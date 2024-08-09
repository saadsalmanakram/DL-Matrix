import tfx
from tfx.components import ExampleGen, StatisticsGen, SchemaGen
from tfx.orchestration.experimental.interactive.interactive_context import InteractiveContext

# Create an InteractiveContext
context = InteractiveContext()

# Define components
example_gen = ExampleGen(input_base='path/to/data')
statistics_gen = StatisticsGen(examples=example_gen.outputs['examples'])
schema_gen = SchemaGen(stats=statistics_gen.outputs['stats'])

# Add components to the context
context.run([example_gen, statistics_gen, schema_gen])
