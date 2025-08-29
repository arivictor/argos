import json

import msgspec

from argos.domain.entity import WorkflowDSL

schema = msgspec.json.schema(WorkflowDSL)
print(json.dumps(schema, indent=2))
