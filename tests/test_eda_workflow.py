import eda_workflow.eda_workflow


def test_eda_workflow_module_imports():
    assert eda_workflow.eda_workflow.WORKFLOW_NAME == "eda_workflow"
