try:    
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    # sys.path.append(osp.join(BASE_DIR, 'external'))
    sys.path.append(osp.join(BASE_DIR, 'external', 'structural_losses'))
    from tf_nndistance import nn_distance
    from tf_approxmatch import approx_match, match_cost
except:
    print('External Losses (Chamfer-EMD) were not loaded.')
