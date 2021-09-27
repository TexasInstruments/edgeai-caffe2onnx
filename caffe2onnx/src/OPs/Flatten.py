import caffe2onnx.src.c2oObject as Node
##---------------------------------------------Softmax--------------------------------------------------------------##
# Get hyperparameters
def getFlattenAttri(layer):
    axis = layer.softmax_param.axis
    # Hyperparameter dictionary
    dict = {"axis": axis}
    return dict

# Calculate the output dimension
def getFlattenOutShape(input_shape, dict):
    # Calculate the output dimensionoutput_shape
    output_shape = input_shape.copy() # same as input dimension
    num_dims = len(output_shape[0])
    axis = dict["axis"]
    out_size = 1
    for i in range(num_dims - axis):
        out_size = out_size*output_shape[0][axis+i]
        output_shape[0][axis+i] = 1
    output_shape[0][num_dims-1] = out_size
    
    return output_shape

# Build node
def createFlatten(layer, nodename, inname, outname, input_shape):
    dict = getFlattenAttri(layer)
    output_shape = getFlattenOutShape(input_shape, dict)
    # Build node
    node = Node.c2oNode(layer, nodename, "Flatten", inname, outname, input_shape, output_shape, dict)
    print(nodename, " node construction completed")
    return node