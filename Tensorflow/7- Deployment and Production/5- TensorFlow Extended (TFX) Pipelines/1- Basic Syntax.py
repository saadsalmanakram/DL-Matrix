import tensorflow as tf
import tfx

# Define your TFX components
example_gen = tfx.components.CsvExampleGen(input_base='data_path')
trainer = tfx.components.Trainer(
    module_file='trainer.py',
    examples=example_gen.outputs['examples'],
    schema=schema_gen.outputs['schema']
)
pusher = tfx.components.Pusher(
    model=trainer.outputs['model'],
    push_destination=tfx.proto.PushDestination(
        filesystem=tfx.proto.PushDestination.Filesystem(
            base_directory='serving_model_directory'
        )
    )
)

# Define the pipeline
pipeline = tfx.dsl.Pipeline(
    pipeline_name='my_pipeline',
    components=[example_gen, trainer, pusher],
    enable_cache=True,
    metadata_connection_config=tfx.orchestration.metadata.sqlite_metadata_connection_config('metadata.db')
)

# Run the pipeline
tfx.orchestration.LocalDagRunner().run(pipeline)
