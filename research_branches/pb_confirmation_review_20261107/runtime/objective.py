from pb_degree import RelationDegreeCDF
def make_objective(case,cfg,dataset):
    return RelationDegreeCDF(case['space'].types,case['space'].pairs,case['prior'],case['clean'],data_sha256=cfg['datasets'][dataset]['sha256'])
