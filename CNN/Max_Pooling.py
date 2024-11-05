import numpy as np

def max_pooling(input_matrix, pool_size, stride):
    # Get the dimension of the input matrix
    height, width = input_matrix.shape
    pool_height, pool_width = pool_size
    # Calculate the output dimensions
    output_height = (height-pool_height)//stride+1
    output_width = (width-pool_width)//stride+1
    # Create an output matrix filled with zeros
    output_matrix = np.zeros((output_height, output_width))
    # Perform Max-Pooling 
    for i in range(output_height):
        for j in range(output_width):
            # Define the current window
            row_start = i*stride
            row_end = row_start + pool_height
            col_start = j*stride
            col_end = col_start + pool_width
            # Extract the window and apply max operation
            window = input_matrix[row_start: row_end, col_start:col_end]
            output_matrix[i,j] = np.max(window)
    return output_matrix



mat = np.array([[1,2,3,0],
                [4,5,6,1],
                [7,8,9,2],
                [3,0,4,1]])
pool_size = (2,2)
stride = 1

print(max_pooling(mat, pool_size, stride))
