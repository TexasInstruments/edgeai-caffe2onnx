import caffe2onnx.src.c2oObject as Node
import copy
##---------------------------------------------Softmax--------------------------------------------------------------##
# Get hyperparameters
def getSoftmaxAttri(layer):
    axis = layer.softmax_param.axis
    # Hyperparameter dictionary
    dict = {"axis": axis}
    return dict

# Calculate the output dimension
def getSoftmaxOutShape(input_shape, dict):
    # Calculate the output dimensionoutput_shape
    output_shape = copy.deepcopy(input_shape) # same as input dimension
    output_shape[0][dict["axis"]] = 1
    
    return output_shape

# Build node
def createArgmax(layer, nodename, inname, outname, input_shape):
    dict = getSoftmaxAttri(layer)
    #print(nodename, "In Shape", input_shape)
    output_shape = getSoftmaxOutShape(input_shape, dict)
    # Build node
    node = Node.c2oNode(layer, nodename, "ArgMax", inname, outname, input_shape, output_shape, dict)
    #print(nodename, " node construction completed", input_shape, output_shape)
    return node