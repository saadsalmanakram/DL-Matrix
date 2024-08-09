<script>
  async function loadModel() {
    const model = await tf.loadLayersModel('https://example.com/model.json');
    const prediction = model.predict(tf.tensor([/* input data */]));
    prediction.print();
  }
  loadModel();
</script>
