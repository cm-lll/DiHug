"""Strict continuation gate. Tolerances never replace exact optimizer state."""
def restoration_passes(record):
    try:
        return (record['optimizer_exact'] is True and record['rng_exact'] is True
            and record['loss']['passes'] is True and len(record['inputs'])==2
            and all(row[key]['passes'] is True for row in record['inputs'] for key in ['logits','p0','loss'])
            and all(record[group] and all(v['passes'] is True for v in record[group].values())
                    for group in ['gradients','parameters','optimizer']))
    except (KeyError,TypeError):return False
