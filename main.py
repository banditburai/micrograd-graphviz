from fasthtml.common import *
from fasthtml.svg import *
from graphviz import Digraph

tailwindLink = Link(rel="stylesheet", href="styleoutput.css", type="text/css")
app, rt = fast_app(
    pico=False,    
    hdrs=(tailwindLink,
          Script(src="https://cdn.jsdelivr.net/npm/svg-pan-zoom@3.6.1/dist/svg-pan-zoom.min.js"))
)

# Micrograd Code taken from https://raw.githubusercontent.com/EurekaLabsAI/micrograd/refs/heads/master/utils.py and
# https://raw.githubusercontent.com/EurekaLabsAI/micrograd/refs/heads/master/micrograd.py
# -----------------------------------------------------------------------------
# rng related
# class that mimics the random interface in Python, fully deterministic,
# and in a way that we also control fully, and can also use in C, etc.

class RNG:
    def __init__(self, seed):
        self.state = seed

    def random_u32(self):
        # xorshift rng: https://en.wikipedia.org/wiki/Xorshift#xorshift.2A
        # doing & 0xFFFFFFFFFFFFFFFF is the same as cast to uint64 in C
        # doing & 0xFFFFFFFF is the same as cast to uint32 in C
        self.state ^= (self.state >> 12) & 0xFFFFFFFFFFFFFFFF
        self.state ^= (self.state << 25) & 0xFFFFFFFFFFFFFFFF
        self.state ^= (self.state >> 27) & 0xFFFFFFFFFFFFFFFF
        return ((self.state * 0x2545F4914F6CDD1D) >> 32) & 0xFFFFFFFF

    def random(self):
        # random float32 in [0, 1)
        return (self.random_u32() >> 8) / 16777216.0

    def uniform(self, a=0.0, b=1.0):
        # random float32 in [a, b)
        return a + (b-a) * self.random()

random = RNG(42)
# -----------------------------------------------------------------------------
# Value. Similar to PyTorch's Tensor but only of size 1 element

class Value:
    """ stores a single scalar value and its gradient """

    def __init__(self, data, _prev=(), _op=''):
        self.data = data
        self.grad = 0
        # internal variables used for autograd graph construction
        self._backward = lambda: None
        self._prev = _prev
        self._op = _op # the op that produced this node, for graphviz / debugging / etc

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), '+')

        def _backward():
            self.grad += out.grad
            other.grad += out.grad
        out._backward = _backward

        return out

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), '*')

        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        out._backward = _backward

        return out

    def __pow__(self, other):
        assert isinstance(other, (int, float)), "only supporting int/float powers for now"
        out = Value(self.data**other, (self,), f'**{other}')

        def _backward():
            self.grad += (other * self.data**(other-1)) * out.grad
        out._backward = _backward

        return out

    def relu(self):
        out = Value(0 if self.data < 0 else self.data, (self,), 'ReLU')

        def _backward():
            self.grad += (out.data > 0) * out.grad
        out._backward = _backward

        return out

    def tanh(self):
        out = Value(math.tanh(self.data), (self,), 'tanh')

        def _backward():
            self.grad += (1 - out.data**2) * out.grad
        out._backward = _backward

        return out

    def exp(self):
        out = Value(math.exp(self.data), (self,), 'exp')

        def _backward():
            self.grad += out.data * out.grad
        out._backward = _backward

        return out

    def log(self):
        # (this is the natural log)
        out = Value(math.log(self.data), (self,), 'log')

        def _backward():
            self.grad += (1/self.data) * out.grad
        out._backward = _backward

        return out

    def backward(self):
        print("backward called")
        # topological order all of the children in the graph
        topo = []
        visited = set()
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)
        build_topo(self)

        # go one variable at a time and apply the chain rule to get its gradient
        self.grad = 1
        for v in reversed(topo):
            v._backward()

    def __neg__(self): # -self
        return self * -1.0

    def __radd__(self, other): # other + self
        return self + other

    def __sub__(self, other): # self - other
        return self + (-other)

    def __rsub__(self, other): # other - self
        return other + (-self)

    def __rmul__(self, other): # other * self
        return self * other

    def __truediv__(self, other): # self / other
        return self * other**-1

    def __rtruediv__(self, other): # other / self
        return other * self**-1

    def __repr__(self):
        return f"Value(data={self.data}, grad={self.grad})"

# -----------------------------------------------------------------------------
# Multi-Layer Perceptron (MLP) network. Module here is similar to PyTorch's nn.Module

class Module:

    def zero_grad(self):
        for p in self.parameters():
            p.grad = 0

    def parameters(self):
        return []

class Neuron(Module):

    def __init__(self, nin, nonlin=True):
        self.w = [Value(random.uniform(-1, 1)) for _ in range(nin)]
        self.b = Value(0)
        self.nonlin = nonlin
        # color the neuron params light green (only used in graphviz visualization)
        # vis_color([self.b] + self.w, "lightgreen")

    def __call__(self, x):
        act = sum((wi*xi for wi,xi in zip(self.w, x)), self.b)
        return act.tanh() if self.nonlin else act

    def parameters(self):
        return self.w + [self.b]

    def __repr__(self):
        return f"{'TanH' if self.nonlin else 'Linear'}Neuron({len(self.w)})"

class Layer(Module):

    def __init__(self, nin, nout, **kwargs):
        self.neurons = [Neuron(nin, **kwargs) for _ in range(nout)]

    def __call__(self, x):
        out = [n(x) for n in self.neurons]
        return out

    def parameters(self):
        return [p for n in self.neurons for p in n.parameters()]

    def __repr__(self):
        return f"Layer of [{', '.join(str(n) for n in self.neurons)}]"

class MLP(Module):

    def __init__(self, nin, nouts):
        sz = [nin] + nouts
        self.layers = [Layer(sz[i], sz[i+1], nonlin=i!=len(nouts)-1) for i in range(len(nouts))]

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]

    def __repr__(self):
        return f"MLP of [{', '.join(str(layer) for layer in self.layers)}]"

# -----------------------------------------------------------------------------
# loss function: the negative log likelihood (NLL) loss
# NLL loss = CrossEntropy loss when the targets are one-hot vectors
# same as PyTorch's F.cross_entropy

def cross_entropy(logits, target):    
    # subtract the max for numerical stability (avoids overflow)
    # commenting these two lines out to get a cleaner visualization
    # max_val = max(val.data for val in logits)
    # logits = [val - max_val for val in logits]
    # 1) evaluate elementwise e^x
    ex = [x.exp() for x in logits]
    # 2) compute the sum of the above
    denom = sum(ex)
    # 3) normalize by the sum to get probabilities
    probs = [x / denom for x in ex]
    # 4) log the probabilities at target
    logp = (probs[target]).log()
    # 5) the negative log likelihood loss (invert so we get a loss - lower is better)
    nll = -logp

    return nll


# -----------------------------------------------------------------------------
# The AdamW optimizer, same as PyTorch optim.AdamW

class AdamW:
    def __init__(self, parameters, lr=1e-1, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.0):
        self.parameters = parameters
        self.lr = lr
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.weight_decay = weight_decay
        # state of the optimizer
        self.t = 0 # step counter
        for p in self.parameters:
            p.m = 0 # first moment
            p.v = 0 # second moment

    def step(self):
        self.t += 1
        for p in self.parameters:
            if p.grad is None:
                continue
            p.m = self.beta1 * p.m + (1 - self.beta1) * p.grad
            p.v = self.beta2 * p.v + (1 - self.beta2) * (p.grad ** 2)
            m_hat = p.m / (1 - self.beta1 ** self.t)
            v_hat = p.v / (1 - self.beta2 ** self.t)
            p.data -= self.lr * (m_hat / (v_hat ** 0.5 + 1e-8) + self.weight_decay * p.data)

    def zero_grad(self):
        for p in self.parameters:
            p.grad = 0

# -----------------------------------------------------------------------------
# data related - generates Yin Yang dataset
# Thank you https://github.com/lkriener/yin_yang_data_set

def gen_data_yinyang(random: RNG, n=1000, r_small=0.1, r_big=0.5):
    pts = []

    def dist_to_right_dot(x, y):
        return ((x - 1.5 * r_big)**2 + (y - r_big)**2) ** 0.5

    def dist_to_left_dot(x, y):
        return ((x - 0.5 * r_big)**2 + (y - r_big)**2) ** 0.5

    def which_class(x, y):
        d_right = dist_to_right_dot(x, y)
        d_left = dist_to_left_dot(x, y)
        criterion1 = d_right <= r_small
        criterion2 = d_left > r_small and d_left <= 0.5 * r_big
        criterion3 = y > r_big and d_right > 0.5 * r_big
        is_yin = criterion1 or criterion2 or criterion3
        is_circles = d_right < r_small or d_left < r_small

        if is_circles:
            return 2
        return 0 if is_yin else 1

    def get_sample(goal_class=None):
        while True:
            x = random.uniform(0, 2 * r_big)
            y = random.uniform(0, 2 * r_big)
            if ((x - r_big)**2 + (y - r_big)**2) ** 0.5 > r_big:
                continue
            c = which_class(x, y)
            if goal_class is None or c == goal_class:
                scaled_x = (x / r_big - 1) * 2
                scaled_y = (y / r_big - 1) * 2
                return [scaled_x, scaled_y, c]

    for i in range(n):
        goal_class = i % 3
        x, y, c = get_sample(goal_class)
        pts.append([[x, y], c])

    tr = pts[:int(0.8 * n)]
    val = pts[int(0.8 * n):int(0.9 * n)]
    te = pts[int(0.9 * n):]
    return tr, val, te

# -----------------------------------------------------------------------------
# Color scheme configuration
# -----------------------------------------------------------------------------
# Select colors from https://www.graphviz.org/doc/info/colors.html#brewer or hex

COLOR_SCHEME = {
    'node': {
        'input': '/pastel28/3',
        'target': '/pastel28/1',
        'weight': '/pastel28/5',
        'bias': '/pastel28/5',
        'neuron': '/pastel28/7',
        'logit': '/pastel28/2',
        'loss_calc': '/prgn9/3',
        'inter_calc': '/pastel28/4',
        'final_loss': '/pastel28/6',
        'unknown': '/pastel28/8'
    },
    'cluster': {
        'input': '/blues9/1',
        'params': '/purples9/3',
        'intermediate': '/greens9/1',
        'output': '/oranges9/2',
        'loss': '/bupu9/3',
        'unknown': '/greys9/1'
    },
    'background': 'transparent',
    'edge': '/greys9/5'
}
# -----------------------------------------------------------------------------
# visualization related functions from utils.py

# Instead of directly setting colors, let's set node types:
def set_node_type(nodes, node_type):
    if not isinstance(nodes, (list, tuple)):
        nodes = [nodes]
    for n in nodes:
        setattr(n, '_node_type', node_type)

def trace(root, node_callback=None):
    nodes, edges = [], []
    def build(v):
        if v not in nodes:
            nodes.append(v)
            if node_callback:
                node_callback(v)
            for child in v._prev:
                if (child, v) not in edges:
                    edges.append((child, v))
                build(child)
    build(root)
    return nodes, edges
# -----------------------------------------------------------------------------

def loss_fun(model, split):
    # evaluate the loss function on a given data split
    total_loss = Value(0.0)
    for x, y in split:
        logits = model(x)
        target = Value(y)  # Create a Value node for the target
        loss = cross_entropy(logits, target)
        total_loss = total_loss + loss
    mean_loss = total_loss * (1.0 / len(split))
    return mean_loss

# -----------------------------------------------------------------------------
# svg-pan-zoom related
panzoomscript = """
function initializeSvgPanZoom() {
    const container = document.getElementById('svg-container');
    const svgElement = container.querySelector('svg');
    if (svgElement) {
        // Set the SVG background to transparent
        svgElement.style.backgroundColor = 'transparent';
        
        const panZoom = svgPanZoom(svgElement, {
            zoomEnabled: true,
            controlIconsEnabled: true,
            fit: true,            
            center: true,
            minZoom: 0.1,
            maxZoom: 10
        });

        // Resize and fit on window resize
        window.addEventListener('resize', function() {
            panZoom.resize();
            panZoom.fit();
            panZoom.center();
        });
    }
}

// Listen for the custom event that will be triggered after the SVG is updated
document.body.addEventListener('svgUpdated', function() {
    document.getElementById('svg-container').classList.remove('loading');
    initializeSvgPanZoom();
});
"""
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
"""
For reference, the attributes that can be set on nodes

VIZ_ATTRIBUTES = {
    'cluster': {'desc': 'Grouping of nodes in viz', 'values': ['cluster_input', 'cluster_params', 'cluster_intermediate', 'cluster_output', 'cluster_loss', 'cluster_unknown']}
}
NODE_ATTRIBUTES = {
    '_node_type': {'desc': 'Role in network', 'values': ['input', 'weight', 'bias', 'neuron', 'logit', 'loss_calc', 'final_loss']},
    '_layer_type': {'desc': 'Layer type', 'values': ['input', 'hidden', 'output', 'loss', 'operation']},
    '_layer_index': {'desc': 'Layer index', 'type': int},
    '_node_index': {'desc': 'Node index in layer', 'type': int},    
    '_prev': {'desc': 'Predecessor nodes', 'type': list},
    '_op': {'desc': 'Operation that produced node', 'type': str}
}
"""
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
import subprocess
import pydot
from collections import defaultdict

def find_subgraphs_by_partial_names(graph_obj, partial_names):
    names_to_find = set(partial_names)
    
    def search_subgraphs(graph):
        found = {}
        for subgraph in graph.get_subgraphs():
            subgraph_name = subgraph.get_name().lower()
            print(f"Found subgraph: {subgraph_name}")
            
            matched_names = {name for name in names_to_find if name.lower() in subgraph_name}
            found.update({name: subgraph for name in matched_names})            
            names_to_find.difference_update(matched_names)            
            if names_to_find:
                found.update(search_subgraphs(subgraph))
            else:
                break
        
        return found

    return search_subgraphs(graph_obj)

def get_required_subgraphs(graph, required_names):
    found_subgraphs = find_subgraphs_by_partial_names(graph, required_names)
    
    if len(found_subgraphs) != len(required_names):
        missing = set(required_names) - set(found_subgraphs.keys())
        print(f"Could not find all required subgraphs. Missing: {list(missing)}")
        return None
    
    return found_subgraphs

def get_bounding_box(cluster):
    for node in cluster.get_nodes():
        if node.get_name() == 'graph':
            bb = node.get('bb')
            if bb:
                return map(float, bb.strip('"').split(','))
    return None

def sort_params_nodes(nodes):
    def sort_key(node):
        layer_type = node.get('layer_type')
        node_index = node.get('node_index')
        if layer_type is None or node_index is None:
            print(f"Warning: Node {node.get_name()} is missing layer_type or node_index")
            return ('', -1)
        return (int(layer_type.split('_')[1]), int(node_index))
    return sorted(nodes, key=sort_key)

def calculate_grid_dimensions(param_nodes, available_width, node_width, h_margin):
    nodes_per_row = max(1, int(available_width / (node_width + h_margin))) - 1 
    num_rows = math.ceil(len(param_nodes) / nodes_per_row)
    num_columns = min(nodes_per_row, len(param_nodes))
    print("Expect num_rows, num_columns:", num_rows, num_columns)
    return num_rows, num_columns

def position_nodes_in_grid(nodes, start_x, start_y, available_width, node_width, node_height, h_margin, v_margin, left_margin):
    num_rows, num_columns = calculate_grid_dimensions(nodes, available_width - left_margin, node_width, h_margin)
    
    x_step = node_width + h_margin
    y_step = node_height + v_margin
    
    highest_y = float('-inf')
    for i, node in enumerate(nodes):
        col = i // num_rows
        row = i % num_rows 
        
        x_position = start_x + (col+1) * x_step
        y_position = start_y + (row+1) * y_step
        
        highest_y = max(highest_y, y_position)
        node.set('pos', f'{x_position},{y_position}!')
    
    return highest_y

def adjust_node_positions(dot_content, num_inputs, num_hidden_layers, num_neurons_per_layer, num_outputs):            
    graphs = pydot.graph_from_dot_data(dot_content)
    if not graphs:
        print("Error: Could not parse DOT content")
        return None
    graph = graphs[0]

    required_subgraphs = ['params', 'intermediate']
    subgraphs = get_required_subgraphs(graph, required_subgraphs)

    if not subgraphs:
        return graph

    params_cluster = subgraphs['params']
    intermediate_cluster = subgraphs['intermediate']
    
    params_bb = get_bounding_box(params_cluster)
    intermediate_bb = get_bounding_box(intermediate_cluster)
    
    if not params_bb or not intermediate_bb:
        print("Could not find bounding boxes")
        return graph
    
    start_params_x, start_params_y, end_params_x, end_params_y = params_bb
    start_inter_x, _, end_inter_x, _ = intermediate_bb = intermediate_bb
    
    available_width = end_params_x - start_params_x

    params_nodes = [node for node in params_cluster.get_nodes() if node.get('node_type') in ['weight', 'bias']]
    
    if len(params_nodes) == 0:
        print("No parameter nodes found in cluster_params.")
        return graph
    
    sorted_params_nodes = sort_params_nodes(params_nodes)
    
    node_width = 100
    node_height = 30 
    h_margin = 50
    v_margin = 20
    left_margin = 0
    params_start_x = start_inter_x + left_margin    

    highest_y = position_nodes_in_grid(sorted_params_nodes, params_start_x, start_params_y, 
                                       available_width, node_width, node_height, h_margin, v_margin, left_margin)
    
    # Adjust bounding box and label position for params
    params_bb_node = next(node for node in params_cluster.get_nodes() if node.get_name() == 'graph')
    params_bb_node.set('bb', f'"{start_inter_x},{start_params_y},{end_inter_x},{highest_y + 50}"')
    
    old_lp = params_bb_node.get('lp').strip('"').split(',')
    if len(old_lp) == 2:
        old_x, old_y = map(float, old_lp)
        new_x = old_x + abs(end_inter_x - end_params_x)
        params_bb_node.set('lp', f'"{new_x},{old_y}"')
    
    return graph

# -----------------------------------------------------------------------------
DEFAULT_ATTRS = {
    "cluster": "cluster_unknown",
    "color": COLOR_SCHEME['node']['unknown'],
    "layer_type": "unknown",
    "node_type": "unknown",
    "layer_index": "?",
    "node_index": "?"
}

def get_node_attrs(n, x, logits, model):
    attrs = defaultdict(lambda key: DEFAULT_ATTRS.get(key, "unknown"))
    
    node_type = getattr(n, '_node_type', 'unknown')
    attrs['node_type'] = node_type
    attrs['color'] = COLOR_SCHEME['node'].get(node_type, COLOR_SCHEME['node']['unknown'])

    match node_type:
        case 'input':
            attrs.update({
                "cluster": "cluster_input",                
                "layer_type": "input",
                "layer_index": "0",
                "node_index": str(x.index(n))
            })
        case 'target':
            attrs.update({
                "cluster": "cluster_input",
                "layer_type": "input",
                "shape": "doublecircle",
                "layer_index": "0",
                "node_index": "0"
            })
        case 'final_loss' | 'loss_calc':
            attrs.update({
                "cluster": "cluster_loss",                
                "layer_type": "loss",
                "layer_index": "final" if node_type == 'final_loss' else "calc",
                "node_index": "0"
            })
        case 'weight' | 'bias':
            for layer_idx, layer in enumerate(model.layers):
                if n in [w for neuron in layer.neurons for w in neuron.w] or n in [neuron.b for neuron in layer.neurons]:
                    attrs.update({
                        "cluster": "cluster_params",                                                
                        "layer_type": f"layer_{layer_idx+1}",
                        "layer_index": str(layer_idx+1),
                        "node_index": str([w for neuron in layer.neurons for w in neuron.w].index(n)) if node_type == 'weight' else str([neuron.b for neuron in layer.neurons].index(n))
                    })
                    break
        case 'neuron':
            for layer_idx, layer in enumerate(model.layers):
                if n in layer.neurons:
                    attrs.update({
                        "cluster": "cluster_intermediate",                        
                        "layer_type": f"layer_{layer_idx+1}",
                        "layer_index": str(layer_idx+1),
                        "node_index": str(layer.neurons.index(n))
                    })
                    break
        case 'logit':
            attrs.update({
                "cluster": "cluster_output",                
                "layer_type": "output",
                "layer_index": str(len(model.layers)),
                "node_index": str(logits.index(n))
            })
        case 'inter_calc':
            attrs.update({
                "cluster": "cluster_intermediate",                
                "layer_type": "inter_calc",
                "layer_index": "calc",
                "node_index": "0"
            })
        case _:  # unknown
            attrs.update({
                "cluster": "cluster_unknown",                
                "layer_type": "unknown",
                "layer_index": "?",
                "node_index": "?"
            })
    
    return dict(attrs) 

# -----------------------------------------------------------------------------
def mark_loss_nodes(loss, logits):
    nodes_to_mark = set()
    def dfs(node):
        if node not in nodes_to_mark and node not in logits:
            nodes_to_mark.add(node)
            for child in node._prev:
                dfs(child)
    dfs(loss)
    for node in nodes_to_mark:
        if node == loss:
            set_node_type(node, 'final_loss')
        else:
            set_node_type(node, 'loss_calc')

def mark_model_nodes(model, x, logits, target):
    set_node_type(target, "target")
    set_node_type(logits, "logit")
    set_node_type(x, "input")
    for layer in model.layers:
        for neuron in layer.neurons:
            set_node_type(neuron, "neuron")
            for w in neuron.w:
                set_node_type(w, "weight")
            set_node_type(neuron.b, "bias")

def mark_intermediate_nodes(node):
    if not hasattr(node, '_node_type'):
        if hasattr(node, '_op'):
            set_node_type(node, 'inter_calc')
        else:
            set_node_type(node, 'unknown')

def mark_node_types(node):
    if not hasattr(node, '_node_type'):
        mark_intermediate_nodes(node)

def create_node_label(n, attrs):
    data_grad = f"data: {n.data:.4f}\ngrad: {n.grad:.4f}"
    
    if attrs['cluster'] in ['cluster_params', 'cluster_output']:
        type_info = f"{attrs['layer_type']} {attrs['node_type']}"
    else:
        type_info = attrs['node_type']
    
    return f"{data_grad}\n{type_info}"

def create_graphviz_svg(model, x, y, num_inputs, num_hidden_layers, num_neurons_per_layer, num_outputs):    
    logits = model(x)
    loss = cross_entropy(logits, y)    
    loss.backward()         
    target = Value(y)
        
    mark_loss_nodes(loss, logits)
    mark_model_nodes(model, x, logits, target)
    
    nodes, edges = trace(loss, mark_node_types)
    nodes.append(target)
    edges.append((target, loss))

    dot = Digraph(format='svg', engine='dot', graph_attr={
        'bgcolor': COLOR_SCHEME['background'],
        'rankdir': 'LR',
        'nodesep': '0.1',
        'ranksep': '0.2',
        'esep': '0.2',
        'compound': 'true',
        'clusterrank': 'local',
        'newrank': 'true', 
        'ratio': '.4',
        'outputmode': "edgesfirst",      
        'splines': 'true',        
    })

    # dot.unflatten(stagger=3, fanout=True)
    # tested unflatten but no effect seen, also tested groups but no effect seen

    # Create clusters for different parts of the network
    with dot.subgraph(name='cluster_input') as input_cluster:
        input_cluster.attr(label='Input', fontsize="30", style='filled', labeljust='l', color=COLOR_SCHEME['cluster']['input'], rank='source')

    with dot.subgraph(name='cluster_params') as params_cluster:
        params_cluster.attr(label='Parameter Nodes', labelloc='b', labeljust='r', fontsize="30", style='filled', color=COLOR_SCHEME['cluster']['params'])

    with dot.subgraph(name='cluster_intermediate') as intermediate_cluster:
        intermediate_cluster.attr(label='Intermediate Nodes', labelloc='b', labeljust='r', fontsize="30", style='filled', color=COLOR_SCHEME['cluster']['intermediate'])

    with dot.subgraph(name='cluster_output') as output_cluster:
        output_cluster.attr(label='Output', fontsize="30", labeljust='l', style='filled', color=COLOR_SCHEME['cluster']['output'])

    with dot.subgraph(name='cluster_loss') as loss_cluster:
        loss_cluster.attr(label='Loss', fontsize="30", labeljust='l', style='filled', color=COLOR_SCHEME['cluster']['loss'])
   
    for n in nodes:
        attrs = get_node_attrs(n, x, logits, model)
        node_id = str(id(n))
        label = create_node_label(n, attrs)
        # label = f"data: {n.data:.4f}\ngrad: {n.grad:.4f}\n{attrs['layer_type']}\n{attrs['node_type']}\nL{attrs['layer_index']}N{attrs['node_index']}"
        fillcolor = attrs['color']
        
        node_attrs = {
            'label': label,
            'shape': 'box',
            'style': 'filled',
            'fillcolor': fillcolor,
            'width': '0.1',
            'height': '0.1',
            'fontsize': '16'
        }
        node_attrs.update(attrs)

        cluster_name = attrs.get('cluster', 'cluster_unknown')
        with dot.subgraph(name=cluster_name) as cluster:
            cluster.node(name=node_id, **node_attrs)

        if n._op:
            op_label = f"{n._op}"
            # op_label = f"{n._op}\n{attrs.get('layer_type', 'unknown')}\nop\nL{attrs.get('layer_index', '?')}N{attrs.get('node_index', '?')}"
            with dot.subgraph(name=cluster_name) as cluster:
                cluster.node(name=node_id + n._op, 
                             label=op_label, 
                             shape='ellipse', 
                             style='filled', 
                             fillcolor='white', 
                             width='0.1', 
                             height='0.1', 
                             fontsize='16',
                             layer_type='operation',
                             node_type='op')
            dot.edge(node_id + n._op, node_id, minlen='1')

    for n1, n2 in edges:
        dot.edge(str(id(n1)), str(id(n2)) + (n2._op if n2._op else ''), minlen='1')

     # Use dot to get the initial layout
    process = subprocess.Popen(['dot', '-Tdot'], 
                               stdin=subprocess.PIPE, 
                               stdout=subprocess.PIPE, 
                               stderr=subprocess.PIPE,
                               text=True)
    dot_content, stderr = process.communicate(input=dot.source)

    if process.returncode != 0:
        print(f"Error in dot: {stderr}")
        return None    

    # Adjust node positions
    adjusted_graph = adjust_node_positions(dot_content, num_inputs, num_hidden_layers, num_neurons_per_layer, num_outputs)

    adjusted_graph.set_prog(['neato', '-n'])

    # Generate SVG
    svg_string = adjusted_graph.create(format='svg').decode('utf-8')

    # Parse the SVG string into a FastHTML SVG component
    svg_content = NotStr(svg_string)
    
    return Svg(svg_content, width="100%", height="100%", 
               viewBox="0 0 1500 1000")


@rt('/')
def get():
    return Title("Micrograd Visualizer"), Div(
        Div(
            H1("Micrograd Visualizer", cls="text-3xl font-bold text-gray-800 mb-4 md:mb-0"),
            Div(
                Form(
                    Div(
                        Div(
                            Label("Inputs:", fr="inputs", cls="block text-sm font-medium text-gray-700"),
                            Input(type="number", name="inputs", id="inputs", value="2", min="1", max="10", cls="input-field"),
                            cls="input-wrapper"
                        ),
                        Div(
                            Label("Hidden Layers:", fr="hidden_layers", cls="block text-sm font-medium text-gray-700"),
                            Input(type="number", name="hidden_layers", id="hidden_layers", value="1", min="1", max="5", cls="input-field"),
                            cls="input-wrapper"
                        ),
                        Div(
                            Label("Neurons per Hidden Layer:", fr="neurons_per_layer", cls="block text-sm font-medium text-gray-700"),
                            Input(type="number", name="neurons_per_layer", id="neurons_per_layer", value="8", min="1", max="20", cls="input-field"),
                            cls="input-wrapper"
                        ),
                        Div(
                            Label("Outputs:", fr="outputs", cls="block text-sm font-medium text-gray-700"),
                            Input(type="number", name="outputs", id="outputs", value="3", min="1", max="10", cls="input-field"),
                            cls="input-wrapper"
                        ),
                        Div(
                        Button(
                            Span("Update", id="buttonText"),
                            type="button", 
                            id="updateButton",
                            hx_post="/update_graph",
                            hx_target="#svg-container",                            
                            hx_trigger="load, click",
                            hx_include="[name='inputs'], [name='hidden_layers'], [name='neurons_per_layer'], [name='outputs']",                            
                            cls="btn"
                        ),
                        cls="button-container"
                    ),
                        cls="input-container"
                    ),
                    
                    cls="w-full h-full"
                ),
                cls="form-container"
            ),
            cls="main-container"
        ),
        Div(
            id="svg-container",
            cls="svg-container loading"
        ),        
        Script(panzoomscript),
        Script("""
             htmx.on("htmx:beforeRequest", function(event) {
                var button = htmx.find("#updateButton");
                button.innerHTML = "Updating...";
                button.disabled = true;
                document.getElementById('svg-container').classList.add('loading');
            });
            htmx.on("htmx:afterRequest", function(event) {
                var button = htmx.find("#updateButton");
                button.innerHTML = "Update";
                button.disabled = false;
            });
        """),        
        cls="flex flex-col p-8 min-h-screen w-full"
    )

@rt('/update_graph')
def post(inputs: int, hidden_layers: int, neurons_per_layer: int, outputs: int):
    hidden_layers_array = [neurons_per_layer] * hidden_layers
    
    # Create a model and input
    model = MLP(inputs, hidden_layers_array + [outputs])
    x = [Value(0.0) for _ in range(inputs)]
    y = 0  # Dummy target

    svg_content = create_graphviz_svg(model, x, y, inputs, hidden_layers, neurons_per_layer, outputs)

    return Div(
        svg_content,
        Script("document.body.dispatchEvent(new Event('svgUpdated'));"),
        Span("Update", id="buttonText", hx_swap_oob="true"),
        style="width: 100%; height: 100%;"
    )

# Run the app
if __name__ == "__main__":
    serve(port=8080)