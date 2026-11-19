from joint_degree import JointRelationDegreeCDF
def make_objective(case,cfg,dataset):
    return JointRelationDegreeCDF(case['space'].types,case['space'].pairs,case['prior'],case['clean'],data_sha256=cfg['datasets'][dataset]['sha256'])
