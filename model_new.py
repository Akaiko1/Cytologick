import config

if config.FRAMEWORK.lower() == 'pytorch':
    from clogic import ai_pytorch as ai
else:
    from clogic import ai


if __name__ == '__main__':
    if config.FRAMEWORK.lower() == 'pytorch':
        ai.train_new_model_pytorch('_new', 3, 5, batch_size=4)
    else:
        ai.train_new_model('_new', 3, 5, batch_size=4)
