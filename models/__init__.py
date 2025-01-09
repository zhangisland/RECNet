from loguru import logger


def create_model(opt):
    # image restoration
    model = opt['model']
    if model == 'rdp':
        from .SIEN_model import SIEN_Model as M
    else:
        raise NotImplementedError('Model [{:s}] not recognized.'.format(model))
    m = M(opt)
    logger.info(f'Model [{m.__class__.__name__}] is created.')
    return m

