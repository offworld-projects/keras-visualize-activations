import keras.backend as K
import numpy as np
import pdb
import matplotlib.pyplot as plt

def get_activations(model, model_inputs, verbose=False, print_shape_only=False, layer_names=None):
    activations = []
    inp = model.input

    model_multi_inputs_cond = True
    if not isinstance(inp, list):
        # only one input! let's wrap it in a list.
        inp = [inp]
        model_multi_inputs_cond = False

    outputs = [layer.output for layer in model.layers if
               layer.name in layer_names or layer_names is None]  # all layer outputs

    funcs = [K.function(inp + [K.learning_phase()], [out]) for out in outputs]  # evaluation functions

    if model_multi_inputs_cond:
        list_inputs = []
        list_inputs.extend(model_inputs)
        list_inputs.append(0.)
    else:
        list_inputs = [model_inputs, 0.]

    # Learning phase. 0 = Test mode (no dropout or batch normalization)
    # layer_outputs = [func([model_inputs, 0.])[0] for func in funcs]
    layer_outputs = [func(list_inputs)[0] for func in funcs]
    if verbose: print('----- activations -----')
    for layer_activations in layer_outputs:
        activations.append(layer_activations)
        if verbose:
            if print_shape_only:
                print(layer_activations.shape)
            else:
                print(layer_activations)
    return activations


def display_activations(activation_maps):
    import matplotlib.pyplot as plt
    """
    (1, 26, 26, 32)
    (1, 24, 24, 64)
    (1, 12, 12, 64)
    (1, 12, 12, 64)
    (1, 9216)
    (1, 128)
    (1, 128)
    (1, 10)
    """
    batch_size = activation_maps[0].shape[0]
    assert batch_size == 1, 'One image at a time to visualize.'
    for i, activation_map in enumerate(activation_maps):
        print('Displaying activation map {}'.format(i))
        shape = activation_map.shape
        if len(shape) == 4:
            activations = np.hstack(np.transpose(activation_map[0], (2, 0, 1)))
        elif len(shape) == 2:
            # try to make it square as much as possible. we can skip some activations.
            activations = activation_map[0]
            num_activations = len(activations)
            if num_activations > 1024:  # too hard to display it on the screen.
                square_param = int(np.floor(np.sqrt(num_activations)))
                activations = activations[0: square_param * square_param]
                activations = np.reshape(activations, (square_param, square_param))
            else:
                activations = np.expand_dims(activations, axis=0)
        else:
            #raise Exception('len(shape) = 3 has not been implemented.')
            pass
        plt.imshow(activations, interpolation='None', cmap='jet')
        plt.show()


def stack_activations(activation_maps):
    # stack all activations onto one numpy array
    actvs = []
    cmap = plt.get_cmap('jet') # jet heatmap

    batch_size = activation_maps[0].shape[0]
    assert batch_size == 1, 'One image at a time to visualize.'
    for i, activation_map in enumerate(activation_maps):
        shape = activation_map.shape
        if len(shape) == 4: # (1xWxHxC)
            activations = np.hstack(np.transpose(activation_map[0], (2, 0, 1)))
        elif len(shape) == 2: # (1xW)
            # try to make it square as much as possible. we can skip some activations.
            activations = activation_map[0]
            num_activations = len(activations)
            if num_activations > 1024:  # too hard to display it on the screen.
                square_param = int(np.floor(np.sqrt(num_activations)))
                activations = activations[0: square_param * square_param]
                activations = np.reshape(activations, (square_param, square_param))
            else:
                activations = np.expand_dims(activations, axis=0)
        else:
            #raise Exception('len(shape) = 3 has not been implemented.')
            pass
        actv = cmap(activations) #convert to JET before adding to the rest
        actvs.append( actv )

    # convert separate actvs into one
    height = np.sum([ps.shape[0] for ps in actvs]) + len(actvs)+1 # total sum and some margin
    width = np.max([ps.shape[1] for ps in actvs])
    outarr = np.ones((height, width, 4))*np.nan # 3-channel

    hvalue = 0
    for actv in actvs:
        outarr[hvalue:hvalue+actv.shape[0], 0:actv.shape[1], :] = actv # copy into outarr
        hvalue += actv.shape[0]+1

    return outarr, actvs # order: h, w