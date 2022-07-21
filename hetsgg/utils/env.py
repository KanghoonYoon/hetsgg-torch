import os

from hetsgg.utils.imports import import_file


def setup_environment():

    custom_module_path = os.environ.get("TORCH_DETECTRON_ENV_MODULE")
    if custom_module_path:
        setup_custom_environment(custom_module_path)
    else:
        pass


def setup_custom_environment(custom_module_path):

    module = import_file("hetsgg.utils.env.custom_module", custom_module_path)
    assert hasattr(module, "setup_environment") and callable(
        module.setup_environment
    ), (
        "Custom environment module defined in {} does not have the "
        "required callable attribute 'setup_environment'."
    ).format(
        custom_module_path
    )
    module.setup_environment()


setup_environment()
