import neptune.new as neptune

id_prefix_sqgans = "THES-"
id_prefix_wqgans = "THES2-"


def get_ids(pure_ids, prefix):
    return [f"{prefix}{el}" for el in pure_ids]


def get_data_for_id(rid, project):
    return neptune.init(
        project=f'wiktor.jurasz/{project}',
        api_token=None,  # put the token in NEPTUNE_API_TOKEN env variable
        run=rid)
