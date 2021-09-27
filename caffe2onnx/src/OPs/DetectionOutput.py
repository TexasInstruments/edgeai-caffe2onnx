import caffe2onnx.src.c2oObject as Node
##---------------------------------------------Softmax--------------------------------------------------------------##
# Get hyperparameters


# Build node
def createDetectionOutput(layer, nodename, inname, outname, input_shape):
    output_shape = [[1000,5],[1000]]
    outname = ["Boxes", "Lables"]
    # Build node
    node = Node.c2oNode(layer, nodename, "Gather", inname, outname, input_shape, output_shape)
    print(nodename, " node construction completed")
    return node