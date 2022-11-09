# encoding: utf-8

from .CNN import CNN
from .loss import get_criterion
from .Transformer import build_transformer, build_transformer_local
from .backbones.vit_pytorch import vit_base_patch16_224_TransReID, vit_small_patch16_224_TransReID, deit_small_patch16_224_TransReID

__factory_T_type = {
    'vit_base_patch16_224_TransReID': vit_base_patch16_224_TransReID,
    'deit_base_patch16_224_TransReID': vit_base_patch16_224_TransReID,
    'vit_small_patch16_224_TransReID': vit_small_patch16_224_TransReID,
    'deit_small_patch16_224_TransReID': deit_small_patch16_224_TransReID
}


def make_model(cfg, logger, num_class, camera_num=0, view_num=0):
    if cfg.MODEL.BACKBONE == 'transformer':
        if cfg.MODEL.JPM:
            model = build_transformer_local(num_class, camera_num, view_num, cfg, __factory_T_type,
                                            cfg.MODEL.RE_ARRANGE, logger)
            logger.info('===========building transformer with JPM module ===========')
        else:
            model = build_transformer(num_class, camera_num, view_num, cfg, __factory_T_type, logger)
            logger.info('===========building transformer===========')
    elif cfg.MODEL.BACKBONE == 'cnn':
        model = CNN(num_class, cfg, logger)
        logger.info('===========building ResNet===========')
    else:
        raise SyntaxWarning
    return model
