import config

if config.FRAMEWORK.lower() == 'pytorch':
    from clogic import ai_pytorch as ai
else:
    from clogic import ai


if __name__ == '__main__':
    if config.FRAMEWORK.lower() == 'pytorch':
        ai.train_current_model_pytorch('_new', 50, batch_size=64)
    else:
        ai.train_current_model('_new', 50, batch_size=64)
