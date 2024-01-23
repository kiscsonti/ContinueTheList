def filter_none(element):
    return True


def filter_url(element):
    if isinstance(element, str) and "http" in element:
        return True
    return False
