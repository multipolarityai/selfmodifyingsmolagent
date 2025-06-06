print("DEBUG: Script execution started.")
import os
import json
import inspect
import types  # Add this import for proper method binding
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass
from smolagents import CodeAgent, InferenceClientModel
from smolagents.tools import Tool
import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd


@dataclass
class WorkflowNode:
    """Represents a node in the workflow graph"""
    id: str
    name: str
    tool_name: str
    inputs: List[str]
    outputs: List[str]
    parameters: Dict[str, Any]


class DynamicTool(Tool):
    """A dynamically created tool that can be added to the agent"""
    
    def __init__(self, name: str, description: str, code: str, inputs: Dict[str, Dict[str, Any]] = None, output_type: str = "string"):
        # Set required class attributes before calling super().__init__()
        self.name = name
        self.description = description
        self.inputs = inputs or {}
        self.output_type = output_type
        self.code = code
        
        # Extract function from code
        local_vars = {}
        exec(self.code, globals(), local_vars)
        self._function = next(v for v in local_vars.values() if callable(v))
        
        # Before calling super().__init__, we need to dynamically create a proper forward method
        self._create_forward_method()
        
        super().__init__()
    
    def _create_forward_method(self):
        """Dynamically creates a forward method with the correct parameter signature"""
        # Get the parameter names from inputs
        input_params = list(self.inputs.keys())
        
        # For core tools, handle them specifically
        if self.name == "create_tool":
            def create_tool_forward(self, name, description, code):
                return self._function(name=name, description=description, code=code)
            self.forward = types.MethodType(create_tool_forward, self)
        elif self.name == "modify_workflow":
            def modify_workflow_forward(self, action, node_data=None, edge_data=None):
                return self._function(action=action, node_data=node_data, edge_data=edge_data)
            self.forward = types.MethodType(modify_workflow_forward, self)
        elif self.name == "execute_workflow":
            def execute_workflow_forward(self, inputs=None):
                return self._function(inputs=inputs)
            self.forward = types.MethodType(execute_workflow_forward, self)
        else:
            # For dynamically created tools, create a forward method with exact parameter names
            if len(input_params) == 0:
                def forward_method(self):
                    return self._function()
                self.forward = types.MethodType(forward_method, self)
            else:
                # Create parameter string for function definition
                param_str = ', '.join(input_params)
                args_str = ', '.join(input_params)
                
                # Use exec to create a function with the exact signature
                func_def = f"""
def forward_method(self, {param_str}):
    return self._function({args_str})
"""
                local_scope = {}
                exec(func_def, {'self': self}, local_scope)
                forward_method = local_scope['forward_method']
                
                # Bind the method to this instance
                self.forward = types.MethodType(forward_method, self)


class WorkflowGraph:
    """Manages the workflow graph structure"""
    
    def __init__(self):
        self.graph = nx.DiGraph()
        self.nodes: Dict[str, WorkflowNode] = {}
    
    def add_node(self, node: WorkflowNode):
        """Add a node to the workflow graph"""
        self.nodes[node.id] = node
        self.graph.add_node(node.id, **node.__dict__)
    
    def add_edge(self, from_node: str, to_node: str):
        """Add an edge between two nodes"""
        if from_node in self.nodes and to_node in self.nodes:
            self.graph.add_edge(from_node, to_node)
    
    def remove_node(self, node_id: str):
        """Remove a node from the workflow"""
        if node_id in self.nodes:
            self.graph.remove_node(node_id)
            del self.nodes[node_id]
    
    def get_execution_order(self) -> List[str]:
        """Get topological order for workflow execution"""
        try:
            return list(nx.topological_sort(self.graph))
        except nx.NetworkXError:
            return list(self.nodes.keys())  # Fallback if graph has cycles
    
    def visualize(self, save_path: str = "workflow_graph.png"):
        """Visualize the workflow graph"""
        plt.figure(figsize=(12, 8))
        pos = nx.spring_layout(self.graph)
        nx.draw(self.graph, pos, with_labels=True, node_color='lightblue', 
                node_size=1500, font_size=10, font_weight='bold')
        plt.title("Workflow Graph")
        plt.savefig(save_path)
        plt.close()


class SelfModifyingAgent:
    """An LLM agent that can modify its own code and workflow with error correction"""
    
    def __init__(self, model_name: str = "gpt-4", executor_type: str = "e2b", 
                 use_managed_agents: bool = False, hf_token: str = None, 
                 e2b_api_key: str = None):
        print(f"DEBUG: SelfModifyingAgent.__init__() called with executor_type='{executor_type}', use_managed_agents={use_managed_agents}")
        """
        Initialize the self-modifying agent with proper sandbox configuration
        
        Args:
            model_name: Model to use for the agent
            executor_type: Type of executor ('e2b', 'docker', or 'local')
            use_managed_agents: Whether to support multi-agent functionality
            hf_token: HuggingFace token for model access
            e2b_api_key: E2B API key for sandbox access
        """
        self.use_managed_agents = use_managed_agents
        self.executor_type = executor_type
        
        # Validate E2B requirements
        if executor_type == "e2b":
            self.e2b_api_key = e2b_api_key or os.getenv("E2B_API_KEY")
            if not self.e2b_api_key:
                raise ValueError("E2B_API_KEY is required when using E2B executor. "
                               "Set the environment variable or pass e2b_api_key parameter.")
        
        # Initialize model with proper token handling
        if hf_token or os.getenv("HF_TOKEN"):
            model = InferenceClientModel(
                token=hf_token or os.getenv("HF_TOKEN"),
                provider="together"  # or your preferred provider
            )
        else:
            model = InferenceClientModel()
        
        # Initialize the base agent with secure sandbox execution
        if use_managed_agents and executor_type == "e2b":
            # For multi-agent setups, we need to run everything in sandbox
            self.agent = None  # Will be initialized in sandbox
            self._setup_e2b_multiagent()
        else:
            # Simple single-agent setup with sandboxed code execution
            self.agent = CodeAgent(
                tools=[],
                model=model,
                executor_type=executor_type,
                name="self_modifying_agent",
                description="A self-modifying agent that can create tools and workflows"
            )
        
        # Dynamic components
        self.dynamic_tools: Dict[str, DynamicTool] = {}
        self.workflow = WorkflowGraph()
        self.agent_code_history: List[Dict[str, Any]] = []
        
        # Add error tracking for self-correction
        self.error_history: List[Dict[str, Any]] = []
        self.max_retry_attempts = 3
        
        # Add core self-modification tools
        if not use_managed_agents:
            self._add_core_tools()
    
    def _setup_e2b_multiagent(self):
        """Setup E2B sandbox for multi-agent functionality"""
        try:
            from e2b_code_interpreter import Sandbox
            
            self.sandbox = Sandbox(api_key=self.e2b_api_key)
            
            # Install required packages in sandbox
            self.sandbox.commands.run("pip install smolagents networkx matplotlib pandas")
            
            # Create agent initialization code for sandbox
            self.agent_init_code = f"""
import os
from smolagents import CodeAgent, InferenceClientModel

# Initialize the main agent
main_agent = CodeAgent(
    model=InferenceClientModel(token=os.getenv("HF_TOKEN"), provider="together"),
    tools=[],
    name="self_modifying_agent",
    description="A self-modifying agent that can create tools and workflows"
)

# Store reference for later use
_global_agent = main_agent
"""
            
        except ImportError:
            raise ImportError("E2B package not found. Install with: pip install 'smolagents[e2b]'")
        except Exception as e:
            raise RuntimeError(f"Failed to initialize E2B sandbox. Check your E2B_API_KEY: {e}")
    
    def _run_code_in_sandbox(self, code: str, verbose: bool = False) -> str:
        """Execute code in E2B sandbox with error handling"""
        if not hasattr(self, 'sandbox'):
            raise RuntimeError("Sandbox not initialized")
            
        execution = self.sandbox.run_code(
            code,
            envs={
                'HF_TOKEN': os.getenv('HF_TOKEN'),
                'E2B_API_KEY': os.getenv('E2B_API_KEY')
            }
        )
        
        if execution.error:
            execution_logs = "\n".join([str(log) for log in execution.logs.stdout])
            logs = execution_logs + execution.error.traceback
            raise ValueError(f"Sandbox execution error: {logs}")
        
        return "\n".join([str(log) for log in execution.logs.stdout])
    
    def _add_core_tools(self):
        print("DEBUG: SelfModifyingAgent._add_core_tools() called.")
        """Add core tools for self-modification"""
        
        # Tool to create new tools
        create_tool_code = '''
def create_new_tool(name, description, code):
    """Create a new dynamic tool"""
    # This should actually create the tool, not just return a dictionary
    try:
        # Access the parent agent instance to create the tool
        # Note: This will be handled by the forward method binding
        success = True
        result = {
            'action': 'create_tool',
            'name': name,
            'description': description,
            'code': code,
            'success': success
        }
        return result
    except Exception as e:
        return {
            'action': 'create_tool',
            'name': name,
            'description': description,
            'code': code,
            'success': False,
            'error': str(e)
        }
'''
        
        class CreateToolDynamic(DynamicTool):
            def __init__(self, parent_agent, *args, **kwargs):
                self.parent_agent = parent_agent
                super().__init__(*args, **kwargs)
            
            def _create_forward_method(self):
                def create_tool_forward(self, name, description, code):
                    success = self.parent_agent.create_tool(name, description, code)
                    return {
                        'action': 'create_tool',
                        'name': name,
                        'description': description,
                        'code': code,
                        'success': success
                    }
                self.forward = types.MethodType(create_tool_forward, self)
    
        create_tool_instance = CreateToolDynamic(
            self,
            name="create_tool",
            description="Create a new tool dynamically",
            code=create_tool_code,
            inputs={
                "name": {"type": "string", "description": "Name of the tool to create"},
                "description": {"type": "string", "description": "Description of the tool"},
                "code": {"type": "string", "description": "Python code for the tool"}
            },
            output_type="object"
        )
        self._register_tool(create_tool_instance)
        
        # Tool to modify workflow
        modify_workflow_code = '''
def modify_workflow(action, node_data=None, edge_data=None):
    """Modify the workflow graph"""
    result = {
        'action': 'modify_workflow',
        'operation': action,
        'success': True
    }
    
    if action == 'add_node' and node_data:
        result['node_data'] = node_data
    elif action == 'add_edge' and edge_data:
        result['edge_data'] = edge_data
    elif action == 'remove_node' and node_data:
        result['node_id'] = node_data.get('id')
    
    return result
'''
        
        class ModifyWorkflowDynamic(DynamicTool):
            def __init__(self, parent_agent, *args, **kwargs):
                self.parent_agent = parent_agent
                super().__init__(*args, **kwargs)
            
            def _create_forward_method(self):
                def modify_workflow_forward(self, action, node_data=None, edge_data=None):
                    try:
                        if action == "add_node" and node_data:
                            if 'name' not in node_data and 'tool_name' in node_data:
                                node_data['name'] = node_data['tool_name'] + '_node'
                            elif 'name' not in node_data:
                                node_data['name'] = 'unknown_node'
                            success = self.parent_agent.modify_workflow_graph(action, **node_data)
                        elif action == "add_edge" and edge_data:
                            success = self.parent_agent.modify_workflow_graph(
                                action, 
                                from_node=edge_data.get('source'),
                                to_node=edge_data.get('target')
                            )
                        elif action == "remove_node" and node_data:
                            success = self.parent_agent.modify_workflow_graph(
                                action,
                                node_id=node_data.get('id') or node_data.get('name')
                            )
                        else:
                            success = False
                            error_msg = f"Invalid action '{action}' or missing data"
                            self.parent_agent._track_error("modify_workflow_forward", error_msg, {
                                "action": action, "node_data": node_data, "edge_data": edge_data
                            })
                        return {'action': 'modify_workflow', 'operation': action, 'success': success, 'node_data': node_data, 'edge_data': edge_data}
                    except Exception as e:
                        self.parent_agent._track_error("modify_workflow_forward", str(e), {"action": action, "node_data": node_data, "edge_data": edge_data})
                        return {'action': 'modify_workflow', 'operation': action, 'success': False, 'error': str(e), 'node_data': node_data, 'edge_data': edge_data}
                self.forward = types.MethodType(modify_workflow_forward, self)

        modify_workflow_instance = ModifyWorkflowDynamic(
            self,
            name="modify_workflow",
            description="Modify the workflow graph structure",
            code=modify_workflow_code,
            inputs={
                "action": {"type": "string", "description": "Action to perform (add_node, add_edge, remove_node)"},
                "node_data": {"type": "object", "description": "Node data for add/remove operations", "nullable": True},
                "edge_data": {"type": "object", "description": "Edge data for add_edge operation", "nullable": True}
            },
            output_type="object"
        )
        self._register_tool(modify_workflow_instance)
        
        # Tool to execute workflow
        execute_workflow_code = '''
def execute_workflow(inputs=None):
    """Execute the current workflow"""
    result = {
        'action': 'execute_workflow',
        'inputs': inputs or {},
        'success': True,
        'outputs': {}
    }
    return result
'''
    
        class ExecuteWorkflowDynamic(DynamicTool):
            def __init__(self, parent_agent, *args, **kwargs):
                self.parent_agent = parent_agent
                super().__init__(*args, **kwargs)
            
            def _create_forward_method(self):
                def execute_workflow_forward(self, inputs=None):
                    try:
                        outputs = self.parent_agent.execute_workflow(inputs)
                        return {'action': 'execute_workflow', 'inputs': inputs or {}, 'success': True, 'outputs': outputs}
                    except Exception as e:
                        return {'action': 'execute_workflow', 'inputs': inputs or {}, 'success': False, 'error': str(e), 'outputs': {}}
                self.forward = types.MethodType(execute_workflow_forward, self)

        execute_workflow_instance = ExecuteWorkflowDynamic(
            self,
            name="execute_workflow",
            description="Execute the current workflow",
            code=execute_workflow_code,
            inputs={
                "inputs": {"type": "object", "description": "Input data for workflow execution", "nullable": True}
            },
            output_type="object"
        )
        self._register_tool(execute_workflow_instance)

    def _register_tool(self, tool: DynamicTool):
        """Register a tool with the agent"""
        self.dynamic_tools[tool.name] = tool
        if self.agent:
            # Fix: Check if tools is a list, if not, convert or access properly
            if hasattr(self.agent, 'tools'):
                if isinstance(self.agent.tools, list):
                    self.agent.tools.append(tool)
                elif isinstance(self.agent.tools, dict):
                    self.agent.tools[tool.name] = tool
                else:
                    # If tools exists but is neither list nor dict, create a new list
                    self.agent.tools = [tool]
            else:
                # If tools doesn't exist as an attribute, create it
                self.agent.tools = [tool]
    
    def create_tool(self, name: str, description: str, code: str, inputs: Dict[str, Dict[str, Any]] = None, output_type: str = "string") -> bool:
        """Create and register a new dynamic tool with enhanced error handling"""
        try:
            # Ensure common imports are present and fix regex patterns
            common_imports = [
                "import re", "import json", "import os", "import math",
                "from typing import Dict, List, Any, Optional",
                "from collections import Counter"
            ]
            
            # Fix regex escape sequences - use raw string to avoid warning
            fixed_code = code.replace(r'[^\w\s\]', r'[^\w\s]')
            
            # Only add imports that aren't already in the code
            missing_imports = [imp for imp in common_imports 
                              if imp not in fixed_code and imp.split()[1] not in fixed_code]
            
            if missing_imports:
                fixed_code = "\n".join(missing_imports) + "\n\n" + fixed_code
            
            # Enhanced auto-infer inputs if not provided
            if inputs is None:
                import ast
                try:
                    tree = ast.parse(fixed_code)
                    for node in ast.walk(tree):
                        if isinstance(node, ast.FunctionDef):
                            params = [arg.arg for arg in node.args.args if arg.arg != 'self']
                            # Create proper input specification with types
                            inputs = {}
                            for param in params:
                                # Try to infer type from parameter name
                                if param in ['a', 'b', 'num', 'number', 'count', 'value']:
                                    inputs[param] = {"type": "number", "description": f"Numeric input {param}"}
                                elif param in ['text', 'string', 'str', 'message']:
                                    inputs[param] = {"type": "string", "description": f"Text input {param}"}
                                else:
                                    inputs[param] = {"type": "string", "description": f"Input parameter {param}"}
                            break
                    if inputs is None:
                        inputs = {}
                except Exception as parse_error:
                    self._track_error("create_tool_parse", str(parse_error), {"code": fixed_code})
                    inputs = {}
            
            # Create the tool with proper inputs
            new_tool = DynamicTool(name, description, fixed_code, inputs, output_type)
            
            # Test compilation
            compile(fixed_code, f"<tool_{name}>", "exec")
            
            # Register the tool
            self._register_tool(new_tool)
            print(f"Successfully created tool: {name} with inputs: {list(inputs.keys())}")
            return True
            
        except Exception as e:
            self._track_error("create_tool", str(e), {
                "name": name,
                "inputs": inputs,
                "code_length": len(code)
            })
            print(f"Error creating tool {name}: {e}")
            return False
    
    def modify_workflow_graph(self, action: str, **kwargs) -> bool:
        """Modify the workflow graph with enhanced error handling"""
        try:
            if action == "add_node":
                # Ensure all required parameters are present
                required_params = ['id', 'name', 'tool_name', 'inputs', 'outputs', 'parameters']
                for param in required_params:
                    if param not in kwargs:
                        if param == 'id':
                            kwargs['id'] = kwargs.get('name', 'unknown_node')
                        elif param == 'name':
                            kwargs['name'] = kwargs.get('tool_name', 'unknown')
                        elif param == 'inputs':
                            kwargs['inputs'] = ['text']
                        elif param == 'outputs':
                            kwargs['outputs'] = ['result']
                        elif param == 'parameters':
                            kwargs['parameters'] = {}
                
                node = WorkflowNode(**kwargs)
                self.workflow.add_node(node)
                print(f"Successfully added workflow node: {node.name}")
                
            elif action == "add_edge":
                self.workflow.add_edge(kwargs['from_node'], kwargs['to_node'])
                print(f"Successfully added edge: {kwargs['from_node']} -> {kwargs['to_node']}")
                
            elif action == "remove_node":
                self.workflow.remove_node(kwargs['node_id'])
                print(f"Successfully removed node: {kwargs['node_id']}")
                
            return True
            
        except Exception as e:
            self._track_error("modify_workflow", str(e), {
                "action": action,
                "kwargs": kwargs
            })
            print(f"Error modifying workflow: {e}")
            return False
    
    def _track_error(self, operation: str, error: str, context: Dict[str, Any] = None):
        """Track errors for learning and self-correction"""
        error_entry = {
            'timestamp': str(pd.Timestamp.now()),
            'operation': operation,
            'error': error,
            'context': context or {},
        }
        self.error_history.append(error_entry)
        print(f"Error tracked: {operation} - {error}")
    
    def _get_error_patterns(self) -> str:
        """Analyze recent errors to provide correction guidance"""
        if not self.error_history:
            return ""
        
        recent_errors = self.error_history[-5:]  # Last 5 errors
        error_summary = []
        
        for error in recent_errors:
            error_summary.append(f"- {error['operation']}: {error['error']}")
        
        return f"""RECENT ERRORS TO AVOID:
{chr(10).join(error_summary)}

CORRECTION GUIDELINES:
1. When creating tools, always specify inputs parameter with proper parameter names
2. When modifying workflow, ensure all required fields (id, name, tool_name) are provided  
3. Use proper parameter names that match the function signature
4. Test regex patterns and escape sequences properly - use raw strings like r'[^\\\\w\\\\s]'
"""

    def self_modify_with_retry(self, modification_request: str) -> str:
        """Self-modify with retry logic and error correction"""
        for attempt in range(self.max_retry_attempts):
            try:
                print(f"\n=== Modification Attempt {attempt + 1} ===")
                
                # Build enhanced prompt with error context
                error_context = self._get_error_patterns()
                
                prompt = f"""
You are a self-modifying AI agent with error correction capabilities.

Current tools: {list(self.dynamic_tools.keys())}
Current workflow nodes: {list(self.workflow.nodes.keys())}
Execution type: {self.executor_type}

Request: {modification_request}

{error_context}

CRITICAL INSTRUCTIONS:
1. When calling create_tool, you MUST specify the function parameters correctly
2. When calling modify_workflow with add_node, you MUST include: name, tool_name, inputs, outputs
3. Always use raw strings for regex patterns: r'[^\\\\w\\\\s]' (note the double backslashes)
4. Test your code before submitting

CORRECT EXAMPLES:

create_tool(
    name="word_counter",
    description="Count word frequencies in text",
    code='''
import re
from collections import Counter

def word_counter(text):
    text = text.lower()
    text = re.sub(r'[^\\\\w\\\\s]', '', text)
    words = text.split()
    return dict(Counter(words))
'''
)

modify_workflow(
    action="add_node",
    node_data={{
        "name": "word_counter_node",
        "tool_name": "word_counter", 
        "inputs": ["text"],
        "outputs": ["word_frequencies"]
    }}
)

Please implement the requested modifications carefully.
"""
                
                response = self.agent.run(prompt)
                
                # Check if any new tools were actually created
                current_tools = list(self.dynamic_tools.keys())
                current_nodes = len(self.workflow.nodes)
                
                # Store the modification
                self.agent_code_history.append({
                    'request': modification_request,
                    'response': response,
                    'attempt': attempt + 1,
                    'tools_after': current_tools,
                    'nodes_after': current_nodes,
                    'timestamp': str(pd.Timestamp.now()),
                    'execution_environment': self.executor_type
                })
                
                # Check if modification was successful
                if len(current_tools) > 3 or current_nodes > 0:  # More than base tools or any workflow nodes
                    print(f"✅ Modification successful on attempt {attempt + 1}")
                    return response
                else:
                    print(f"❌ Attempt {attempt + 1} failed - no new tools or nodes created")
                    if attempt < self.max_retry_attempts - 1:
                        print("Retrying with enhanced error context...")
                        continue
                
            except Exception as e:
                self._track_error("self_modify", str(e), {"attempt": attempt + 1})
                print(f"❌ Attempt {attempt + 1} failed with exception: {e}")
                if attempt < self.max_retry_attempts - 1:
                    continue
        
        return f"Failed to complete modification after {self.max_retry_attempts} attempts. Check error history."

    def self_modify(self, modification_request: str) -> str:
        """Use the agent to modify itself based on a request"""
        if self.use_managed_agents and hasattr(self, 'sandbox'):
            # Multi-agent sandbox modification
            prompt = f"""
You are a self-modifying AI agent running in a secure sandbox environment.

Current tools: {list(self.dynamic_tools.keys())}
Current workflow nodes: {list(self.workflow.nodes.keys())}
Execution type: {self.executor_type}

Request: {modification_request}

To fulfill this request, you should:
1. Analyze what new capabilities are needed
2. Create any necessary new tools using the create_tool function
3. Modify the workflow graph using modify_workflow if needed
4. Test the modifications safely

Available core functions:
- create_tool(name, description, code): Create a new tool
- modify_workflow(action, node_data, edge_data): Modify workflow
- execute_workflow(inputs): Execute current workflow

IMPORTANT: When creating tools, always include all necessary imports at the beginning of your code.
Common imports that might be needed: import re, import json, import os, import pandas as pd, import numpy as np, from collections import Counter
When using regex patterns, use raw strings (r'pattern') to avoid escape sequence warnings.

All code execution will be sandboxed for security.
Please implement the requested modifications.
"""
            
            # Execute in sandbox
            code_to_run = f"print('Starting modification request...')\n{prompt}"
            result = self._run_code_in_sandbox(code_to_run)
            
            # Store the modification in history
            self.agent_code_history.append({
                'request': modification_request,
                'response': result,
                'timestamp': str(pd.Timestamp.now()),
                'execution_environment': 'e2b_sandbox'
            })
            
            return result
        else:
            # Use retry logic for standard modifications
            return self.self_modify_with_retry(modification_request)

    def get_capabilities(self) -> Dict[str, Any]:
        """Get current agent capabilities including error tracking"""
        return {
            'tools': list(self.dynamic_tools.keys()),
            'workflow_nodes': len(self.workflow.nodes),
            'workflow_edges': self.workflow.graph.number_of_edges(),
            'modification_history_length': len(self.agent_code_history),
            'error_history_length': len(self.error_history),
            'executor_type': self.executor_type,
            'supports_managed_agents': self.use_managed_agents
        }

    def execute_workflow(self, inputs: Dict[str, Any] = None) -> Dict[str, Any]:
        """Execute the current workflow"""
        inputs = inputs or {}
        results = {}
        
        execution_order = self.workflow.get_execution_order()
        
        for node_id in execution_order:
            node = self.workflow.nodes[node_id]
            
            # Prepare inputs for this node
            node_inputs = {}
            for input_name in node.inputs:
                if input_name in inputs:
                    node_inputs[input_name] = inputs[input_name]
                elif input_name in results:
                    node_inputs[input_name] = results[input_name]
            
            # Execute the tool
            if node.tool_name in self.dynamic_tools:
                tool = self.dynamic_tools[node.tool_name]
                node_result = tool.forward(**node_inputs, **node.parameters)
                
                # Store outputs
                for output_name in node.outputs:
                    if isinstance(node_result, dict) and output_name in node_result:
                        results[output_name] = node_result[output_name]
                    else:
                        results[output_name] = node_result
        
        return results

    def debug_last_errors(self, count: int = 5) -> List[Dict[str, Any]]:
        """Get the last N errors for debugging"""
        return self.error_history[-count:] if self.error_history else []

    def reset_error_history(self):
        """Clear error history"""
        self.error_history.clear()
        print("Error history cleared")

    def save_state(self, filepath: str):
        """Save the current agent state"""
        state = {
            'dynamic_tools': {
                name: {
                    'name': tool.name,
                    'description': tool.description,
                    'code': tool.code,
                    'inputs': tool.inputs,
                    'output_type': tool.output_type
                }
                for name, tool in self.dynamic_tools.items()
            },
            'workflow_nodes': {
                node_id: {
                    'id': node.id,
                    'name': node.name,
                    'tool_name': node.tool_name,
                    'inputs': node.inputs,
                    'outputs': node.outputs,
                    'parameters': node.parameters
                }
                for node_id, node in self.workflow.nodes.items()
            },
            'workflow_edges': list(self.workflow.graph.edges()),
            'modification_history': self.agent_code_history,
            'executor_type': self.executor_type,
            'use_managed_agents': self.use_managed_agents
        }
        
        with open(filepath, 'w') as f:
            json.dump(state, f, indent=2)

    def load_state(self, filepath: str):
        """Load a previously saved agent state"""
        with open(filepath, 'r') as f:
            state = json.load(f)
        
        # Restore dynamic tools
        for tool_data in state['dynamic_tools'].values():
            self.create_tool(
                tool_data['name'],
                tool_data['description'],
                tool_data['code'],
                tool_data.get('inputs', {}),
                tool_data.get('output_type', 'string')
            )
        
        # Restore workflow nodes
        for node_data in state['workflow_nodes'].values():
            node = WorkflowNode(**node_data)
            self.workflow.add_node(node)
        
        # Restore workflow edges
        for from_node, to_node in state['workflow_edges']:
            self.workflow.add_edge(from_node, to_node)
        
        # Restore modification history
        self.agent_code_history = state.get('modification_history', [])


def demo_self_modifying_agent():
    print("DEBUG: demo_self_modifying_agent() called.")
    """Demonstrate the enhanced self-modifying agent with error correction"""
    
    print("=== Enhanced Self-Modifying Agent Demo ===")
    
    # Ensure E2B_API_KEY is set if using executor_type="e2b"
    # For local testing, you can use executor_type="local"
    # Ensure HF_TOKEN is set if your model requires it
    
    try:
        agent = SelfModifyingAgent(
            executor_type="local",  # or "e2b" if you have E2B_API_KEY set
            use_managed_agents=False
            # Add e2b_api_key=os.getenv("E2B_API_KEY") if using e2b
            # Add hf_token=os.getenv("HF_TOKEN") if needed
        )
        print("DEBUG: SelfModifyingAgent instantiated.")

        # Example interaction:
        print("Initial capabilities:", agent.get_capabilities())
        
        # Try a simple modification request
        response = agent.self_modify("Create a tool to solve third degree equations.")
        print(f"\nModification response: {response}")
        print(f"Final capabilities: {agent.get_capabilities()}")

        # Show any errors that occurred
        errors = agent.debug_last_errors()
        if errors:
            print(f"\nErrors encountered: {len(errors)}")
            for i, error in enumerate(errors):
                print(f"{i+1}. {error['operation']}: {error['error']}")

        return agent # Return the agent instance if needed elsewhere
        
    except ValueError as ve:
        print(f"DEBUG: ValueError during agent instantiation or setup: {ve}")
        import traceback
        traceback.print_exc()
    except ImportError as ie:
        print(f"DEBUG: ImportError during agent instantiation or setup: {ie}")
        import traceback
        traceback.print_exc()
    except Exception as e:
        print(f"DEBUG: An unexpected error occurred in demo_self_modifying_agent: {e}")
        import traceback
        traceback.print_exc()
    
    return None # Return None if agent creation failed


if __name__ == "__main__":
    print("DEBUG: __main__ block entered.")
    try:
        agent_instance = demo_self_modifying_agent()
        if agent_instance:
            print("DEBUG: demo_self_modifying_agent finished successfully and returned an agent instance.")
        else:
            print("DEBUG: demo_self_modifying_agent finished, but agent instantiation might have failed.")
    except Exception as e:
        print(f"DEBUG: Error in __main__: {e}")
        import traceback
        traceback.print_exc()
