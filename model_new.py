from config import load_config


if __name__ == '__main__':
    cfg = load_config()
    if str(cfg.FRAMEWORK).lower() != 'pytorch':
        raise RuntimeError('TensorFlow training is deprecated; set FRAMEWORK=pytorch')

    from clogic import ai_pytorch as ai
    ai.train_new_model_pytorch(cfg, '_new', 3, 50, batch_size=4)
