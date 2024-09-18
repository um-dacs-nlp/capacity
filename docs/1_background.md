# Background for the internship

## Introduction

Large Language Models (LLMs) have revolutionized natural language processing by demonstrating remarkable abilities in tasks ranging from text generation to question answering. Traditionally, these models have been studied from a mathematical perspective, focusing on their architectures and the theoretical underpinnings that enable their performance. However, an alternative viewpoint considers LLMs through the lens of ethology—the study of animal behavior—by analyzing how they respond to different prompts and environments. While this approach can be educational, it often lacks objectivity and reproducibility. Therefore, a more systematic investigation involving controlled and repeatable experiments is essential for a deeper understanding of LLM capacities.

From a mathematical standpoint, recent studies have begun to quantify the memorization capabilities of transformer-based models. Kim et al. (2023) demonstrated that transformers could memorize $O(d + n + \sqrt(n \cdot N))$ parameters. Similarly, Mahdavi et al. (2024) explored the memorization capacity of multi-head attention transformers, providing insights into how model architecture influences the ability to store and recall information. These findings are crucial for developing models that are both efficient and effective in handling large datasets.

To measure the capacity of LLMs, two primary methods have been employed:

1. **Maximum Library Size (MLS) Method**: This approach focuses on determining the largest library size that the network can fully memorize. The model is trained to memorize every item in the input vector library, and the measurement is based on the maximum library size achievable without loss of information.
2. **Maximum Attainable Capacity (MAC) Method**: In this method, the model is trained on a large library, and the goal is to ascertain the maximum number of samples the network can memorize. This provides a practical limit to the amount of information the model can retain.

Understanding the physical aspects of LLMs has also been a subject of recent research. Allen-Zhu et al. (2023-2024) conducted a series of studies examining the physics of LLMs using highly controlled datasets, such as context-free grammars and structured biographies. One of their significant findings is the importance of data manipulation during training. They introduced the concept of **mixed training**, which involves:

- **Knowledge Augmentation**: Rewriting the pretraining data using small auxiliary models to enhance the information content.
- **Early Incorporation of Instruction Fine-Tuning Data**: Integrating instruction fine-tuning data into the pretraining phase, as adding it too late (during standard fine-tuning) is often ineffective.

These strategies contrast with the traditional sequential approach of pretraining followed by fine-tuning, suggesting that early and integrated data manipulation can lead to better performance.

Another critical insight from Allen-Zhu et al.'s work is the inherent limitation of LLMs in functioning as databases. Their studies indicate that language models cannot perform parameterized searches akin to database queries (e.g., SQL `WHERE` clauses) unless the knowledge is explicitly presented inversely in the training data. This finding underscores the limitations of LLMs in tasks requiring precise data retrieval, emphasizing that they are not substitutes for structured databases.

In terms of model efficiency, scaling laws have been identified as pivotal. A key objective, often referred to as the "holy grail," is achieving a capacity of two bits per parameter. Quantization techniques have been explored to approach and even surpass this theoretical limit. For instance, reducing the numerical precision from 16-bit integers (int16) to 8-bit integers (int8) can potentially exceed the capacity limit, allowing models to store more information per parameter. However, further reduction to 4-bit integers (int4) has been observed to decrease the capacity to approximately 0.7 bits per parameter, highlighting a trade-off between model size and memorization capacity.

The present study aims to investigate the capacity possibilities of transformers with varying architectures, focusing on different types of embeddings—such as absolute, relative, and rotary—and attention mechanisms, including position-based and uniform attention. A critical aspect of this research is to determine where exactly the data is stored within the transformer model. To achieve this, we employ probing and freezing techniques:

- **Probing**: Assessing different layers of the model to extract "knowledge" by evaluating their predictive capabilities.
- **Freezing**: Fixing certain layers during training to observe how capacity changes, thereby identifying layers crucial for memorization.

Understanding memorization in LLMs is not merely about storing information but also about the ability to extract and manipulate that information effectively. While prior research, such as that by Allen-Zhu et al., emphasizes the importance of both memorization and manipulation, our current focus is on the foundational aspect of memorization—specifically, the model's ability to predict the next word or fact accurately.

In summary, this study seeks to deepen the understanding of transformer-based LLMs by exploring their capacity limits and the internal mechanisms that facilitate memorization. By examining different architectural configurations and employing targeted experimental techniques, we aim to contribute valuable insights that could inform the development of more efficient and capable language models.

## References

- Kim, J., et al. (2023). *Provable Memorization Capacity of Transformers*.
- Mahdavi, S., et al. (2024). *Memorization Capacity of Multi-Head Attention Transformers*.
- Allen-Zhu, Z., et al. (2023a).
- Allen-Zhu, Z., et al. (2024a).
- Allen-Zhu, Z., et al. (2024b).
