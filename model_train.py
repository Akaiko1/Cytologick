import os
import glob

from config import load_config


if __name__ == '__main__':
    cfg = load_config()
    model_path = '_new'
    
    if str(cfg.FRAMEWORK).lower() == 'pytorch':
        from clogic import ai_pytorch as ai
        # Continue training if any checkpoint exists for the given base prefix.
        candidates = [
            f'{model_path}_last.pth',
            f'{model_path}_best.pth',
            f'{model_path}_final.pth',
        ]
        epoch_candidates = sorted(glob.glob(f'{model_path}_epoch*.pth'))
        if epoch_candidates:
            candidates.append(epoch_candidates[-1])

        resume_path = next((p for p in candidates if os.path.exists(p)), None)
        if resume_path:
            print(f'Continuing training from {resume_path} (saving under {model_path}_*.pth)')
            ai.train_current_model_pytorch(cfg, resume_path, 50, batch_size=64, save_base_path=model_path)
        else:
            print(f'Training new model, will save to {model_path}')
            ai.train_new_model_pytorch(cfg, model_path, cfg.CLASSES, epochs=50, batch_size=64)
    else:
        raise RuntimeError('TensorFlow training is deprecated; set FRAMEWORK=pytorch')

