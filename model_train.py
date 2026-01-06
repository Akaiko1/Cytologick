import os
import config

if config.FRAMEWORK.lower() == 'pytorch':
    from clogic import ai_pytorch as ai
else:
    from clogic import ai


if __name__ == '__main__':
    model_path = '_new'
    
    if config.FRAMEWORK.lower() == 'pytorch':
        # Check if model already exists for continuing training
        pth_path = f'{model_path}.pth' if not model_path.endswith('.pth') else model_path
        if os.path.exists(pth_path) or os.path.exists(model_path):
            print(f'Continuing training from {model_path}')
            ai.train_current_model_pytorch(pth_path if os.path.exists(pth_path) else model_path, 50, batch_size=64)
        else:
            print(f'Training new model, will save to {model_path}')
            ai.train_new_model_pytorch(model_path, config.CLASSES, epochs=50, batch_size=64)
    else:
        ai.train_current_model('_new', 50, batch_size=64)

