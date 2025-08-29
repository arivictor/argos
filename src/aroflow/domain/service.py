from aroflow.domain.entity import WorkflowDSL


def validate_workflow(data: WorkflowDSL) -> bool:
    """
    Validates the workflow structure and contents.

    :param data: The WorkflowDSL instance to validate
    :type data: WorkflowDSL
    :returns: True if the workflow is valid, raises ValueError otherwise
    :rtype: bool
    :raises ValueError: If the workflow structure or contents are invalid
    """
    if not data.steps:
        raise ValueError("Workflow has no steps")
    seen_ids = set()
    for step in data.steps:
        if step.id in seen_ids:
            raise ValueError(f"Duplicate step id found: {step.id}")
        seen_ids.add(step.id)
        step.validate()
    return True
