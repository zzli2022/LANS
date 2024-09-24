from .fusion import MMFusionBlock

def get_fusion(params, *args):
    mm_fusion = MMFusionBlock(params, *args)
    return mm_fusion