class CollectionSchema:
    def __init__(self, fields, **kwargs):
        self.fields = list(fields) if fields else []
        self.kwargs = kwargs

    def add_field(self, field_name=None, datatype=None, is_primary=False, **kwargs):
        from pymilvus.orm.schema import FieldSchema as FS
        field = FS(name=field_name, dtype=datatype, is_primary=is_primary, **kwargs)
        self.fields.append(field)
        return self


class Function:
    pass
