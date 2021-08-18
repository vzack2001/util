from tensorflow import keras
from pathlib import Path

def save_weights(model: keras.Model, name=None, model_path='temp'):
    """ save model weights to hdf5
        use global `model_path`
    """
    model_name = name or model.name
    filename = Path(model_path).joinpath(f'{model_name}_weights.hdf5')
    print(f'model.save_weights({filename})')
    model.save_weights(filename)
    pass

def load_weights(model: keras.Model, name=None, model_path='temp'):
    """ load model_weights from hdf5
        use global `model_path`
    """
    model_name = name or model.name
    filename = Path(model_path).joinpath(f'{model_name}_weights.hdf5')
    print(f'model.load_weights({filename})')
    model.load_weights(filename)
    pass

def to_json(model: keras.Model, name=None, model_path='temp'):
    """ save model to json config
        use global `model_path`
    """
    model_name = name or model.name
    filename = Path(model_path).joinpath(f'{model_name}.json')
    print(f'model.to_json({filename})')
    json_config = model.to_json()
    #print(json_config)
    with open(filename, mode='w') as f:
        f.write(json_config)
    pass

def from_json(model_name, model_path='temp'):
    """ load keras model from json config
        use global `model_path`
    """
    filename = Path(model_path).joinpath(f'{model_name}.json')
    print(f'keras.models.model_from_json({filename})')
    with open(filename, mode='r') as f:
        json_config = f.read()
    #print(json_config)
    model = keras.models.model_from_json(json_config)
    model.summary()
    return model
