import caffe2onnx.src.c2oObject as Node
import copy
##----------------------------------------------------Dropout-------------------------------------------------------##
# Get hyperparameters
def getDropoutAttri(layer):
    # Drop ratio
    ratio = layer.dropout_param.dropout_ratio
    # Hyperparameter dictionary
    ratio = 0.0

    dict = {"ratio":ratio}
    return dict

def getDropoutOutShape(input_shape):
    #  Calculate the output dimensionoutput_shape
    output_shape = copy.deepcopy(input_shape)  # Same as input dimension
    return output_shape

# Build node
def createDropout(layer, nodename, inname, outname, input_shape):
    dict = getDropoutAttri(layer)
    output_shape = getDropoutOutShape(input_shape)
    #  Build node
    node = Node.c2oNode(layer, nodename, "Dropout", inname, outname, input_shape, output_shape)
    #print(nodename, " node construction completed")
    return node