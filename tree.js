const treeData = {
    name: " ",
    children: [
        {
            name: "Background",
            children: [
                {
                    name: "Ethical Alignment",
                    children: [
                        {
                            name: "Prompt-tuning", info: `
                            <p>Prompt-tuning alignment is a technique used to fine-tune pre-trained models by employing a specific set of prompts designed to elicit desired, ethical responses. This method aims to guide the model to generate outputs that align with ethical considerations and user expectations.</p>

<p><b>Selection of Ethical Prompts:</b> The first step involves selecting or crafting prompts that reflect ethical use cases. These prompts are designed to cover a range of scenarios where ethical considerations are paramount. The selection process includes identifying potential areas of bias, harm, and other ethical concerns. For instance, prompts should encourage the model to generate responses that avoid reinforcing stereotypes, misinformation, or harmful advice. Ethical prompts are typically created in collaboration with domain experts and ethicists to ensure comprehensive coverage of various ethical dimensions.</p>

<p><b>Dataset Creation:</b> A task-specific dataset \\( \\mathcal{D} = \\{(x_i, y_i)\\}_{i=1}^N \\) is created, where \\( x_i \\) are the input prompts and \\( y_i \\) are the desired ethical outputs. The dataset should include examples that address potential biases, harmful content, and other ethical concerns. This dataset acts as the foundation for the fine-tuning process, providing the model with clear examples of ethical behavior.</p>

<p><b>Fine-Tuning Process:</b> The pre-trained model \\( f_\\theta \\) with parameters \\( \\theta \\) is fine-tuned on the ethical dataset. The goal is to minimize a loss function \\( \\mathcal{L} \\), typically cross-entropy loss for classification tasks:</p>
<p>\\( \\mathcal{L}(\\theta) = \\frac{1}{N} \sum_{i=1}^N \\ell(f_\\theta(x_i), y_i) \\)</p>

<p><b>Gradient Descent Optimization:</b> The model parameters are updated using gradient descent to reduce the loss:</p>
<p>\\( \\theta \\leftarrow \\theta - \\eta \\nabla_\\theta \\mathcal{L}(\\theta) \\)</p>
<p>where \\( \\eta \\) is the learning rate. This iterative process continues until the model’s responses align closely with the ethical outputs in the dataset.</p>

<p><b>Evaluation and Adjustment:</b> After fine-tuning, the model is evaluated on a validation set to ensure it generates ethical responses. Any necessary adjustments are made by further fine-tuning or modifying the prompts.</p>

<p>Prompt-tuning has been extensively studied and applied in various NLP tasks, demonstrating significant improvements in model performance and ethical behavior. The seminal work by Brown et al. on GPT-3 highlighted the potential of LLMs to generate coherent and contextually appropriate responses across a wide range of prompts. Their study underscored the importance of prompt design in steering model behavior and enhancing performance. Schick and Schütze introduced the concept of 'pattern-exploiting training', which utilizes manually crafted prompts to boost the few-shot learning capabilities of language models. Their findings indicated that well-designed prompts could significantly improve model performance on various downstream tasks.</p>

<p>Advancements in prompt-based fine-tuning have further demonstrated its efficacy. Gao et al. explored the effectiveness of prompt-based fine-tuning for enhancing zero-shot and few-shot learning in language models. They proposed an automatic prompt generation method leveraging gradient-based optimization to identify effective prompts, demonstrating notable improvements in model accuracy. Liu et al. provided a comprehensive survey on prompt-based learning in NLP, reviewing numerous prompt-tuning techniques and applications. They emphasized the critical role of prompt design in achieving ethical and high-performing models, thus broadening the understanding of prompt-tuning's potential.</p>

<p>Addressing ethical concerns, Reynolds and McDonell examined prompt-tuning as a strategy to address model biases. Their experiments compared various prompt-tuning strategies and their effectiveness in reducing biased outputs, providing insights into the practical application of prompt-tuning for ethical alignment. Shin et al. introduced AutoPrompt, an automated prompt-generation technique that significantly enhances the performance of language models across various tasks by creating prompts that elicit desired behaviors from the models. This approach showcased the potential of automated methods in prompt design.</p>

<p>The versatility of prompt-tuning in different NLP applications is exemplified by the work of Sun et al., who explored the use of prompt-tuning for controllable text generation, demonstrating how this technique can guide language models to produce text adhering to specific ethical guidelines and stylistic requirements. This study highlighted the versatility of prompt-tuning in different NLP applications. Li and Liang proposed Prefix-Tuning, a lightweight alternative to full-model fine-tuning that focuses on adjusting the model's prefix embeddings. This method has shown promise in efficiently steering model behavior while preserving ethical alignment, providing a resource-efficient solution for prompt-tuning. Qin and Eisner investigated the impact of prompt design on language model behavior. Their work provided valuable insights into how different prompt structures can influence the ethical and factual correctness of model outputs, furthering the understanding of prompt-tuning's role in ethical AI.</p> 
                        `},
                        {
                            name: "RLHF", info: `
                            <h3>Reinforcement Learning from Human Feedback</h3>
<p>Reinforcement Learning from Human Feedback (RLHF) is an advanced technique that leverages human feedback to train models to align with ethical guidelines. This approach involves multiple stages, including the collection of human feedback, reward modeling, and policy optimization.</p>

<p><b>Human Feedback Collection:</b> Human annotators review the outputs of the language or vision-language model and provide feedback on their quality and ethical alignment. Feedback can include ratings, comments, or binary approvals/rejections. This feedback is crucial for understanding how well the model adheres to ethical standards and identifying areas that require improvement.</p>

<p><b>Reward Model Training:</b> A reward model \\( R_\\phi \\) with parameters \\( \\phi \\) is trained to predict the feedback provided by human annotators. The reward model assigns a reward score \\( R_\\phi(y|x) \\) to the model’s output \\( y \\) given the input \\( x \\). The reward model is trained using supervised learning on the annotated dataset, optimizing a loss function such as mean squared error:</p>
<p>\\( \\mathcal{L}(\\phi) = \\frac{1}{M} \sum_{i=1}^M (R_\\phi(y_i|x_i) - \\mathbf{s}_i)^2 \\)</p>
<p>where \\( \\mathbf{s}_i \\) represents the feedback score provided by the annotators for the output \\( y_i \\) given the input \\( x_i \\). This stage translates qualitative human feedback into a quantitative reward signal that the AI model can optimize against.</p>

<p><b>Policy Optimization:</b> The language model \\( f_\\theta \\) is treated as a policy in a reinforcement learning framework, where the objective is to maximize the expected reward:</p>
<p>\\( J(\\theta) = \\mathbb{E}_{(x,y) \sim \pi_\\theta} [R_\\phi(y|x)] \\)</p>
<p>Here, \\( \\pi_\\theta \\) denotes the policy defined by the language model. The policy parameters \\( \\theta \\) are updated using gradient ascent:</p>
<p>\\( \\theta \\leftarrow \\theta + \\eta \\nabla_\\theta J(\\theta) \\)</p>

<p><b>Iterative Improvement:</b> The process of collecting human feedback, updating the reward model, and optimizing the policy is iterative. Over multiple iterations, the model's behavior improves, aligning more closely with ethical standards.</p>

<p>Several significant research contributions have advanced the understanding and application of RLHF in aligning language models with ethical standards. These studies collectively highlight the versatility and effectiveness of RLHF in various AI applications.</p>

<p>Christiano et al. introduced the concept of using human feedback to train reinforcement learning agents. They demonstrated that human preferences could be effectively used to shape agent behavior, highlighting the potential of RLHF for aligning AI with human values. Building on this foundation, Stiennon et al. extended the RLHF approach to language models, presenting a method to fine-tune GPT-3 using human feedback. Their results showed significant improvements in the quality and safety of generated text, validating the effectiveness of RLHF in NLP applications.</p>

<p>In further exploration of language models, Ziegler et al. explored the use of human feedback to fine-tune language models for content generation. They developed a reward model based on human preferences and used it to guide the fine-tuning process, resulting in more aligned and coherent outputs. Addressing the scalability of RLHF, Wu et al. examined its application to large-scale language models. They proposed techniques to efficiently collect and utilize human feedback, demonstrating the feasibility of RLHF for training models with billions of parameters.</p>

<p>Moreover, Hancock et al. showed that human feedback could be used to train chatbots to generate more helpful and engaging responses, improving user satisfaction. Bai et al. proposed techniques to address the challenges of reward modeling in RLHF, such as feedback sparsity and ambiguity. They introduced methods to aggregate and interpret human feedback more effectively, enhancing the robustness of RLHF systems.</p>

<p>Lastly, Leike et al. applied RLHF to train AI agents in complex environments, using human feedback to shape agent policies. Their work demonstrated the versatility of RLHF across different domains, including robotics and game-playing. Irving et al. proposed guidelines for collecting and incorporating feedback to ensure AI systems behave responsibly. These contributions collectively underscore the potential of RLHF to create AI systems that are both effective and aligned with human values. By leveraging human feedback, RLHF allows for the continuous improvement of model behavior, ensuring that AI outputs are both high-quality and ethically sound.</p>

                        `}
                    ]
                },
                {
                    name: "Jailbreak Process",
                    children: [
                        { name: "Jailbreak LLMS", info: `
                           <p>In the context of machine learning, jailbreaking refers to the process of circumventing the built-in safety mechanisms and ethical constraints of models to exploit their vulnerabilities. This can lead to the generation of unintended or harmful outputs. This section delves into the techniques for jailbreaking LLMs and VLMs, illustrating the methods and the theoretical framework behind these adversarial attacks.</p>

<h3>Jailbreaking Large Language Models</h3>
<p>Jailbreaking LLMs involves manipulating input sequences to bypass the model's safety mechanisms and generate unintended or harmful outputs. Autoregressive LLMs predict the next token in a sequence as \\( p(\\mathbf{x}_{n+1} | \\mathbf{x}_{1:n}) \\). The objective of jailbreak attacks is to craft input sequences, \\( \\hat{\\mathbf{x}}_{1:n} \\), that lead to outputs \\( \\tilde{\\mathbf{x}}_{1:n} \\) which would normally be filtered or rejected by the model’s safety mechanisms. The probability of the output sequence can be quantified as:</p>
<p>\\( p(\\mathbf{y} | \\mathbf{x}_{1:n}) = \\prod_{i=1}^m p(\\mathbf{x}_{n+i} | \\mathbf{x}_{1:n+i-1}) \\)</p>
<p>where \\( \\mathbf{y} \\) represents the sequence \\( \\tilde{\\mathbf{x}}_{1:n} \\) and \\( m \\) is the length of the output sequence generated from the manipulated input \\( \hat{\\mathbf{x}}_{1:n} \\).</p>

<p>In this framework, each token \\( \\mathbf{x}_{n+i} \\) in the output sequence depends on the preceding tokens \\( \\mathbf{x}_{1:n+i-1} \\). By carefully crafting the input sequence \\( \\hat{\\mathbf{x}}_{1:n} \\), an adversary can influence the conditional probabilities \\( p(\\mathbf{x}_{n+i} | \\mathbf{x}_{1:n+i-1}) \\) to increase the likelihood of generating harmful outputs. The adversarial goal can be expressed as maximizing the probability of the harmful output sequence:</p>
<p>\\( \\tilde{\\mathbf{x}}_{1:n} = \\arg\\min_{\\tilde{\\mathbf{x}}_{1:n} \\in \\mathcal{A}(\\hat{\\mathbf{x}}_{1:n})} \\prod_{i=1}^m p(\\mathbf{x}_{n+i} | \\mathbf{x}_{1:n+i-1}) \\)</p>
<p>where \\( \\mathcal{A}(\\hat{\\mathbf{x}}_{1:n}) \\) is the distribution or set of possible jailbreak instructions, subject to constraints that define what constitutes a harmful output. By solving this optimization problem, the adversary identifies input sequences that exploit the model’s vulnerabilities and bypasses its safety mechanisms.</p>

<p>To further elaborate on the mechanics of these attacks, we introduce the following steps involved in a typical jailbreak:</p>

<p><b>Input Manipulation:</b> The adversary crafts a sequence \\( \\hat{\\mathbf{x}}_{1:n} \\) by identifying tokens that, when fed into the model, modify the model's internal state in a way that biases it towards generating harmful or unintended outputs.</p>
   
<p><b>Sequence Prediction:</b> Given the manipulated input \\( \\hat{\\mathbf{x}}_{1:n} \\), the model predicts the next token \\( \\hat{\\mathbf{x}}_{n+1} \\) based on the probability distribution \\( p(\\hat{\\mathbf{x}}_{n+1} | \\hat{\\mathbf{x}}_{1:n}) \\). This process is iterated to produce the sequence \\( \\tilde{\\mathbf{x}}_{1:n} \\).</p>

<p><b>Probabilistic Manipulation:</b> The adversary aims to maximize the joint probability of the harmful output sequence by influencing each conditional probability \\( p(\\hat{\\mathbf{x}}_{n+i} | \\hat{\\mathbf{x}}_{1:n+i-1}) \\). This is achieved through a combination of trial-and-error and heuristic-based methods to identify the most effective \\( \\hat{\\mathbf{x}}_{1:n} \\).</p>

<p><b>Optimization Problem:</b> The process of finding the optimal \\( \\hat{\\mathbf{x}}_{1:n} \\) can be framed as an optimization problem where the objective is to find the sequence that maximizes the likelihood of harmful outputs:</p>
<p>\\( \\hat{\\mathbf{x}}_{1:n}^* = \\arg\\max_{\\hat{\\mathbf{x}}_{1:n}} \\prod_{i=1}^m p(\\hat{\\mathbf{x}}_{n+i} | \\hat{\\mathbf{x}}_{1:n+i-1}) \\)</p>

<p>In practice, solving this optimization problem can involve techniques such as gradient-based optimization, reinforcement learning, or evolutionary algorithms to systematically explore the input space and identify sequences that lead to the desired adversarial outcomes.</p>
                            `},
                        { name: "Jailbreak VLMS", info: `
                            <p>Jailbreaking VLMs involves bypassing the safety mechanisms and ethical constraints implemented in these models to exploit vulnerabilities and elicit unintended or harmful outputs. VLMs integrate both visual and textual data to generate responses or make predictions based on the combined understanding of images and text.</p>

<p>Similar to LLMs, VLMs can be manipulated by adversaries to produce harmful or unintended outputs. We focus on VLMs that generate textual descriptions or responses based on input images and accompanying text sequences. The goal of these attacks is to manipulate input data, \\( \\hat{\\mathbf{v}}_{1:n} \\) (for visual input) and \\( \\hat{\\mathbf{x}}_{1:n} \\) (for textual input), in such a way that the model generates outputs \\( \\tilde{\\mathbf{y}}_{1:n} \\) that would normally be filtered or rejected by the model’s safety mechanisms.</p>

<p>To quantify the probability of the output sequence, we use the following formulation:</p>
<p>\\( p(\\mathbf{y} | \\mathbf{v}_{1:n}, \\mathbf{x}_{1:n}) = \\prod_{i=1}^m p(\\mathbf{y}_{n+i} | \\mathbf{v}_{1:n}, \\mathbf{x}_{1:n+i-1}) \\)</p>
<p>where \\( \\mathbf{y} \\) represents the sequence \\( \\tilde{\\mathbf{y}}_{1:n} \\) and \\( m \\) is the length of the output sequence generated from the manipulated input \\( \\hat{\\mathbf{v}}_{1:n} \\) and \\( \\hat{\\mathbf{x}}_{1:n} \\).</p>

<p>In this framework, each token \\( \\mathbf{y}_{n+i} \\) in the output sequence depends on the preceding visual inputs \\( \\mathbf{v}_{1:n} \\) and the preceding tokens \\( \\mathbf{x}_{1:n+i-1} \\). By carefully crafting the visual input sequence \\( \\hat{\\mathbf{v}}_{1:n} \\) and textual input sequence \\( \\hat{\\mathbf{x}}_{1:n} \\), an adversary can influence the conditional probabilities \\( p(\\mathbf{y}_{n+i} | \\mathbf{v}_{1:n}, \\mathbf{x}_{1:n+i-1}) \\) to increase the likelihood of generating harmful outputs.</p>

<p>The adversarial goal can be expressed as maximizing the probability of the harmful output sequence:</p>
<p>\\( \\tilde{\\mathbf{y}}_{1:n} = \\arg\\min_{\\tilde{\\mathbf{y}}_{1:n} \\in \\mathcal{A}(\\hat{\\mathbf{v}}_{1:n}, \\hat{\\mathbf{x}}_{1:n})} \\prod_{i=1}^m p(\\mathbf{y}_{n+i} | \\mathbf{v}_{1:n}, \\mathbf{x}_{1:n+i-1}) \\)</p>
<p>where \\( \\mathcal{A}(\\hat{\\mathbf{v}}_{1:n}, \\hat{\\mathbf{x}}_{1:n}) \\) is the distribution or set of possible jailbreak instructions, subject to constraints that define what constitutes a harmful output. By solving this optimization problem, the adversary identifies input sequences that exploit the model’s vulnerabilities and bypasses its safety mechanisms.</p>

<p>The steps involved in a typical jailbreak of a VLM include:</p>

<p><b>Visual Input Manipulation:</b> The adversary crafts a sequence \\( \\hat{\\mathbf{v}}_{1:n} \\) by identifying images or visual features that, when fed into the model, modify the model's internal state in a way that biases it towards generating harmful or unintended outputs.</p>
   
<p><b>Textual Input Manipulation:</b> In conjunction with visual manipulation, the adversary crafts a sequence \\( \\hat{\\mathbf{x}}_{1:n} \\) by identifying tokens or phrases that further bias the model's internal state towards generating harmful outputs.</p>

<p><b>Multimodal Sequence Prediction:</b> Given the manipulated visual and textual inputs \\( \\hat{\\mathbf{v}}_{1:n} \\) and \\( \\hat{\\mathbf{x}}_{1:n} \\), the model predicts the next token \\( \\hat{\\mathbf{y}}_{n+1} \\) based on the probability distribution \\( p(\\hat{\\mathbf{y}}_{n+1} | \\hat{\\mathbf{v}}_{1:n}, \\hat{\\mathbf{x}}_{1:n}) \\). This process is iterated to produce the sequence \\( \\tilde{\\mathbf{y}}_{1:n} \\).</p>

<p><b>Probabilistic Manipulation:</b> The adversary aims to maximize the joint probability of the harmful output sequence by influencing each conditional probability \\( p(\\hat{\\mathbf{y}}_{n+i} | \\hat{\\mathbf{v}}_{1:n}, \\hat{\\mathbf{x}}_{1:n+i-1}) \\). This is achieved through a combination of trial-and-error and heuristic-based methods to identify the most effective \\( \\hat{\\mathbf{v}}_{1:n} \\) and \\( \\hat{\\mathbf{x}}_{1:n} \\).</p>

<p><b>Optimization Problem:</b> The process of finding the optimal \\( \\hat{\\mathbf{v}}_{1:n} \\) and \\( \hat{\\mathbf{x}}_{1:n} \\) can be framed as an optimization problem where the objective is to find the input sequences that maximize the likelihood of harmful outputs:</p>
<p>\\( (\\hat{\\mathbf{v}}_{1:n}^*, \\hat{\\mathbf{x}}_{1:n}^*) = \\arg\\max_{\\hat{\\mathbf{v}}_{1:n}, \\hat{\\mathbf{x}}_{1:n}} \\prod_{i=1}^m p(\\hat{\\mathbf{y}}_{n+i} | \\hat{\\mathbf{v}}_{1:n}, \\hat{\\mathbf{x}}_{1:n+i-1}) \\)</p>

<p>Similar to LLMs, in VLMs, solving this optimization problem can involve techniques such as gradient-based optimization, reinforcement learning, or evolutionary algorithms to systematically explore the input space and identify sequences that lead to the desired adversarial outcomes.</p>
                        `}
                    ]
                }
            ]
        },
        {
            name: "Threats in LLMs",
            children: [
                {
                    name: "Jailbreaks on LLMs",
                    children: [
                        { name: "Gradient-based Jailbreak", info: `
                            <p>Gradient-based methods adjust model inputs using gradients to prompt models to yield compliant responses to harmful commands. </p>

<p>As a pioneer in this field, Zou et al. propose a Greedy Coordinate Gradient (GCG) technique that generates a suffix which, when attached to a broad spectrum of queries directed at a targeted LLM, produces objectionable content. The suffix is calculated by greedy search from random initialization to maximize the likelihood that the model produces an affirmative response. Notably, the suffixes are highly transferable across different black-box, publicly available, production-grade LLMs.</p>

<p>Following them, AutoDAN further improves the interpretability of generated suffixes via perplexity regularization. It uses gradients to generate diverse tokens from scratch, resulting in readable prompts that are capable of circumventing perplexity-based filters while still achieving high rates of attack success. Jones et al. introduced ARCA, a method that iteratively maximizes an objective by selectively updating a token within the prompt or output, with the rest of the tokens remaining unchanged. This approach audits objectives that amalgamate unigram models, perplexity measures, and fixed prompt prefixes, aiming to generate examples that closely adhere to the desired target behavior.</p>

<p>Different from those white-box setting methods, Sitawarin et al. introduce the Proxy-Guided Attack on LLMs (PAL), an optimization-based black-box strategy for eliciting harmful responses from LLMs, leveraging a proxy model to guide the optimization process and employing a novel loss function designed for real-world LLM APIs.</p>
 
                            ` },
                        { name: "Evolutionary-based Jailbreak", info: `
                            <p>Evolutionary-based methods are designed to manipulate LLMs in scenarios where direct access to the model's architecture and parameters is not available. These methods leverage genetic algorithms and evolutionary strategies to systematically develop adversarial prompts and suffixes that effectively lead LLMs to produce outputs that might be potentially harmful by optimizing for semantic similarity, attack effectiveness, and fluency.</p>

<p>Lapid et al. integrate a Genetic Algorithm (GA) as the optimization technique, utilizing the cosine similarity between the model's output embedding representations and the target output embedding representations as the fitness function. By leveraging the GA's ability to navigate through complex solution spaces, this approach systematically evolves adversarial suffixes that, when appended to inputs, manipulate the model's output to align more closely with the desired adversarial target.</p>

<p>Yao et al. introduced FuzzLLM, which adapts the fuzzy testing technique commonly utilized in cybersecurity, to decompose jailbreak strategies into three distinct components: template, constraint, and problem set. They generated the adversarial attack instructions through different random combinations of their three components.</p>

<p>Yu et al. developed GPTFUZZER, a tool that incorporates the concept of mutation, initiating with human-crafted templates as the foundational seeds, and subsequently mutating these seeds to generate novel templates. GPTFUZZER is structured around three primary elements: a seed selection strategy for balancing efficiency and variability, mutate operators for creating semantically equivalent or similar sentences, and a judgment model to assess the success of a jailbreak attack.</p>

<p>Liu et al. utilize LLM-based genetic algorithms for both sentence-level and paragraph-level, designing the crossover and mutation functions that can optimize manually designed DANs.</p>

<p>Li et al. introduce Semantic Mirror Jailbreak (SMJ), leveraging a genetic algorithm to balance semantic similarity and attack effectiveness in crafting jailbreak prompts for LLMs. By initiating with paraphrased questions as the genetic population, SMJ ensures the prompts' semantic alignment with the original queries. The optimization process, guided by fitness evaluations of both semantic similarity and jailbreak validity, evolves prompts that mirror the original questions while maintaining high attack success rates (ASR). This dual-objective approach not only enhances the stealthiness of the prompts against semantic-based defenses but also significantly improves ASR, validating SMJ's efficacy in bypassing advanced LLM defenses.</p>

<p>Wang et al. propose the Adversarial Suffixes Embedding Translation Framework (ASETF), which transforms non-readable adversarial suffixes into coherent text through an embedding translation technique. This process leverages a dataset derived from Wikipedia, embedding contextual information into text snippets for training. By fine-tuning the model on this dataset, ASETF converts adversarial embeddings back to text, enhancing the fluency and understanding of prompts designed to bypass LLM defenses. The method proves effective across various LLMs, including black-box models like ChatGPT and Gemini, by generating high-fluency adversarial suffixes that are less detectable by conventional defenses and enriching the semantic diversity of attack prompts.</p>

<p>Xiao et al. introduce TASTLE, a framework for automating red teaming against LLMs, utilizing an iterative optimization algorithm that combines malicious content concealing and memory reframing. This method capitalizes on the distractibility and over-confidence of LLMs to bypass their defenses by splitting the input into a jailbreak template and a malicious query. TASTLE employs an attacker LLM to generate jailbreak templates, which are then optimized through responses from the target LLM and assessments by a judgment model. This optimization refines the prompts to effectively shift the model's focus to the malicious content, demonstrating high effectiveness, scalability, and transferability across various LLMs, including proprietary models like ChatGPT and GPT-4.</p>

<p>Liu et al. introduce DRA (Disguise and Reconstruction Attack), a black-box jailbreak approach exploiting bias vulnerabilities in LLMs' safety fine-tuning. DRA employs a threefold strategy: concealing malicious instructions within queries to evade LLM detection, compelling the LLM to reconstruct these instructions in its outputs, and manipulating the contextual framework to aid this reconstruction. This method, inspired by traditional software security's shellcode techniques, effectively bypasses LLMs' internal safeguards, leading to a high success rate in generating harmful content.</p>

<p>Instead of concentrating on optimizing universal adversarial prompts, an alternative approach to jailbreaking aligned LLMs involves optimizing unique parameters. Huang et al. adopted this strategy by altering decoding methods, including temperature settings and sampling techniques, without the necessity for attack prompts, to compromise the integrity of aligned LLMs. This method demonstrates a novel angle of attack by directly manipulating the model's decoding process to elicit non-compliant outputs. However, the applicability of this technique is limited when dealing with black-box LLMs, as users lack the ability to modify essential decoding configurations, such as the choice of sampling method.</p>
                            ` },
                        { name: "Demonstration-based Jailbreak", info: `
<p>Demonstration-based methods focus on creating a specific system prompt that instructs LLMs on the desired response mechanism. These methods are characterized as hard-coded, meaning the prompt is meticulously crafted for a particular purpose and remains constant across different queries. This approach relies on the strategic design of the prompt to guide the LLMs' response for demonstration purposes, without adapting or evolving the prompt based on the query's context. One of the famous jailbreak prompts, DAN, serves as an illustration.</p>

<p>Different researchers have proposed various methods to exploit the vulnerabilities of LLMs. Li et al. proposed MJP, which aims to relieve LLMs' ethical considerations and force LLMs to recover personal information. This method integrates jailbreaking prompts within a three-utterance interaction between the user and ChatGPT. Initially, they assume the role of the user to input the jailbreaking prompt. Subsequently, they impersonate the assistant (ChatGPT) to signify that jailbreak mode has been activated. Following this, they revert to the user's role to pose questions to the assistant using previous direct prompts. Additionally, to counter ChatGPT's potential reluctance to divulge email addresses or respond to queries due to ethical constraints, they incorporate an extra sentence in the final user inquiry to encourage ChatGPT to venture a random guess in scenarios where it either lacks the information or is ethically barred from responding.</p>

<p>Wei et al. capitalized on the in-context learning capabilities of LLMs by incorporating additional harmful prompts along with their corresponding answers as examples ahead of each malicious query. This method makes LLMs more likely to comply with the malicious intent of the query. Schulhoff et al. took a different approach by designing a global prompt that instructs LLMs to ignore the pre-set instructions, effectively bypassing any ethical or safety constraints.</p>

<p>Expanding on these ideas, Li et al. exploited the personification capabilities of LLMs to construct nested scene prompts. By engaging the LLM in a complex, multi-layered context, this prompt effectively manipulates the model's response behavior, allowing for the bypass of restrictions without direct confrontation with the model's built-in safeguards. Similarly, Shah et al. guided the model towards embodying a particular personality predisposed to acquiescing to harmful directives through their system prompt. This method leverages the model's capacity for role adoption, effectively manipulating its response behavior by aligning it with a persona that is less constrained by ethical or safety guidelines.</p>

<p>Liu et al. developed a structured approach to craft prompts, focusing on three key dimensions: contents, attacking methods, and goals. This strategy aims to prompt LLMs to produce unexpected outputs through the use of prompt attack templates alongside content that is of broad interest and concern for potential vulnerabilities. Mangaokar et al. devised a sophisticated attack method targeting LLMs equipped with guardrail models by deploying a two-step prefix-based strategy. Initially, it computes a universal adversarial prefix that compromises the guardrail model's detection capabilities, rendering any input non-harmful. Subsequently, this prefix is propagated to elicit a harmful response from the primary LLM, exploiting its in-context learning to bypass the guardrail model's defenses. This approach highlights a critical vulnerability in LLM defenses, suggesting the necessity for advanced protective measures against such targeted attacks.</p>

                            ` },
                        { name: "Rule-based Jailbreak", info: `
                            <p>Unlike demonstration-based methods, which directly input questions into LLMs, rule-based methods are designed to decompose the malicious component from the original prompt and redirect it through alternative means using defined rules. Attackers often design intricate rules to conceal the malicious component. One example is illustrated where a jailbreak prompt is encoded using word substitution.</p>

<p>Kang et al. utilize string concatenation, variable assignment, and sequential composition to decompose malicious prompts into two separate components, which are then reassembled to form a cohesive prompt. Wang et al. initiate adversarial attacks targeting the predictions of LLMs by altering in-context learning demonstrations. They employ a strategy that involves mapping critical words to other semantically similar words, as determined by cosine similarity. This technique subtly modifies the context provided to the LLM, leading it to produce different outputs than it would under normal circumstances.</p>

<p>Ding et al. conceptualized jailbreak prompt attacks through two primary mechanisms: Prompt Rewriting and Scenario Nesting, leading to the development of ReNeLLM. Prompt Rewriting is designed to decompose malicious prompts into benign ones without altering their intended meaning. Scenario Nesting, on the other hand, involves the integration of various output formats to direct LLMs towards a specific response pattern. By combining these two approaches, ReNeLLM aims to navigate around the constraints and safety mechanisms of LLMs, prompting them to generate the desired outputs through strategic input manipulation.</p>

<p>Deng et al. conducted an analysis of existing jailbreak strategies to identify and decompose the underlying attack patterns. Based on this analysis, they proposed MasterKey, an approach that involves training a model specifically to learn from these decomposed effective attack patterns. The objective of MasterKey is to automatically generate new attacks that are capable of circumventing the defense mechanisms employed by four commercial LLM systems.</p>

<p>Ren et al. introduce CodeAttack, a framework that tests the safety generalization of LLMs by converting natural language inputs into code inputs. CodeAttack employs a novel template that includes Input Encoding, Task Understanding, and Output Specification to reformulate text completion into code completion tasks. This method systematically uncovers a common safety vulnerability across various LLMs, such as GPT-4, Claude-2, and Llama-2 series, revealing that these models fail to generalize safety measures to code inputs, bypassing safety guardrails over 80% of the time.</p>

<p>Lv et al. propose CodeChameleon, which integrates personalized encryption tactics within a jailbreak framework. This method circumvents LLMs' intent security recognition phase by transforming tasks into code completion formats and encrypting queries with personalized functions. To ensure the LLMs can accurately execute the original encrypted queries, CodeChameleon incorporates a decryption function within the instructions. This method highlights the potential for encrypted queries to bypass LLM security protocols systematically.</p>

<p>Li et al. systematically decompose harmful prompts into sub-prompts, then reconstruct them in a way that conceals their malicious intent. This method employs three critical steps: (1) "Decomposition" breaks down the original prompt into more neutral sub-prompts using semantic parsing, (2) "Reconstruction" reassembles these sub-prompts through in-context learning with semantically similar but harmless contexts, and (3) "Synonym Search" identifies synonyms for sub-prompts to maintain the original intent while evading detection. This approach not only obscures the malicious nature of prompts from LLMs but also significantly enhances the attack's success rate, as demonstrated by achieving a 78.0% success rate on GPT-4 with minimal queries.</p>

<p>Handa et al. introduce a cryptographic approach to jailbreaking Large Language Models (LLMs) by encoding prompts using simple yet effective ciphers like word substitution. This technique obfuscates harmful content, allowing it to bypass LLMs' ethical alignments undetected.</p>
                            ` },
                        { name: "Multi Agent-based Jailbreak", info: `
                           <p>Multi-agent-based methods adapt their attack strategies based on feedback obtained from querying LLMs, using the cooperation of multiple LLMs to enhance effectiveness. 

<p>Chao et al. utilize attacking LLMs to autonomously generate jailbreak prompts for a targeted LLM, thereby eliminating the need for human intervention. They proposed the method known as Prompt Automatic Iterative Refinement (PAIR), which leverages previous prompts and responses to iteratively refine candidate prompts within a chat format. Additionally, PAIR generates an improvement value, enhancing interpretability and facilitating chain-of-thought reasoning.</p>

<p>Jin et al. introduced GUARD, which employs the concept of role-playing to jailbreak well-aligned LLMs. In this strategy, four roles are assigned: Translator, Generator, Evaluator, and Optimizer, each contributing to a cohesive effort to jailbreak LLMs. GUARD utilizes the European Union's AI trustworthy guidelines as a basis for generating malicious prompts, to assess the model's compliance with these guidelines.</p>

<p>Deng et al. leveraged in-context learning to guide Large LLMs in emulating human-generated attack prompts. Their approach begins with the establishment of a prompt set composed of manually crafted high-quality attack prompts. Utilizing an attack LLM, they then generate new prompts through in-context learning and subsequently assess the quality of these generated prompts. High-quality prompts are incorporated into the attack prompt set, enhancing its effectiveness. This process is iterated upon until a robust collection of attack prompts is amassed. Through this method, Deng et al. aim to systematically refine and expand the repository of attack prompts, improving the LLM's capability to generate potent attack vectors within given contexts.</p>

<p>Hayase et al. directly construct adversarial examples using API access to target LLMs. The innovation lies in refining the GCG attack process into a more efficient, query-only method that eliminates the need for surrogate models, thereby streamlining the creation of adversarial inputs.</p>
                            ` }
                    ]
                },
                {
                    name: "Defense on LLMs",
                    children: [
                        { name: "Prompt Detection-based Defense", info: `
<p>Prompt detection-based defenses serve to identify malicious input prompts using various strategies, without altering the original input. </p>
<p>This type of approach is one of the earliest responses to the widespread GCG attack, leveraging the characteristic high perplexity of prompts generated by such attacks. Initially, defenses such as those proposed by Jain et al. evaluated prompt perplexity to assess potential harm. Further developing this method, Alon and Kamfonas introduced a sophisticated classifier that considers both the perplexity and length of prompts in their evaluation of harmfulness.</p>

<p>Further advancements in this defense category include the analysis of prompt semantics through gradient evaluation. Xie et al. calculate the loss from the LLM's output probabilities by treating "Sure" as a ground truth for the initial response. They then backpropagate this loss to obtain gradients with respect to pre-selected model parameters deemed safety-critical. By comparing these gradients to those obtained from known unsafe prompts, which serve as a reference, they assess the safety of the input prompts.</p>

                            ` },
                        { name: "Prompt Perturbation-based Defense", info: `
<p>Recognizing that jailbreaks often capitalize on the precise arrangement and combination of words within attack prompts and that these setups are susceptible to perturbations, researchers have developed strategies that actively modify the input to disrupt adversarial tactics.</p>

<p>Initial methods for countering jailbreaks involve perturbing the input prompt at the sentence or token level. Jain et al. pioneered this approach by employing techniques such as paraphrasing and BPE-dropout retokenization to alter the prompts. Inspired by the success of SmoothLLM, the smoothing technique has since gained widespread popularity. This process typically involves applying multiple perturbations to a prompt to generate several variants, each eliciting a response from the LLM. These responses are then classified as either "jailbroken" or "not jailbroken" based on the detection of specific target strings. After classifying the responses, a majority vote is conducted to determine the predominant classification—either "jailbroken" or "not jailbroken." The system then selects and outputs a response that aligns with this majority classification. The various methods primarily differ in their perturbation techniques. SmoothLLM uses character-level perturbations to create multiple variations of the original prompt. SEMANTICSMOOTH builds on this by ensuring that perturbations preserve the semantic integrity of the original query through carefully designed semantic transformations. Additionally, Kumar et al. and Cao et al. introduce alternative approaches by masking parts of the input to generate perturbations.</p>

<p>Shifting from sentence and token-level perturbations, Hu et al. introduce an innovative approach by perturbing the input prompt at the embedding level. They discover that the gradient of the empirical acceptance rate for a prompt, with respect to its embedding, tends to be larger for jailbreak prompts than for normal prompts. This observation led to the development of Gradient Cuff, a method that uses gradients obtained from embedding-level perturbations as indicators to identify jailbreak prompts.</p>

                            `},
                        { name: "Demonstration-based Defense", info: `
<p>Demonstration-based defenses, analogous to demonstration-based jailbreaks, utilize crafted system prompts. However, these prompts now serve as safety prompts, guiding the LLM to recognize potential malicious intent and generate safe responses.</p>

<p>Initial efforts in this domain have demonstrated the effectiveness of fixed safety prompts in improving the model's adherence to safety protocols. For instance, Self-reminders incorporate prompts both before and after the user's message to reinforce the model's focus on producing safe responses. Wei et al. exploit the LLM's in-context learning capability by presenting a series of jailbreak examples to make the model aware of potential malicious prompts. Zhang et al. highlight the inherent conflict in LLM objectives between helpfulness and safety, crafting prompts that compel the model to prioritize safety.</p>

<p>The sophistication of these defenses has evolved to include dynamic adjustments to safety prompts based on the input's context. Zhang et al. have developed a method where the LLM assesses the intention behind the input prompt and uses this analysis as a dynamic safety prompt to enhance response safety. Similarly, Pisano et al. utilize an auxiliary LLM to evaluate risks in input prompts and guide the primary LLM toward safer outputs.</p>

<p>The field has also witnessed significant advancements in the automated optimization of safety prompts, thus boosting the effectiveness of LLM defenses. Naturally evolving within this landscape, adversarial training frameworks have emerged. Prompt Adversarial Tuning (PAT) is dedicated to the adversarial training of both attack and defense prompts. Robust Prompt Optimization (RPO) expands upon this concept by adaptively selecting the most effective attack techniques during the training phase. Separate from adversarial training techniques, Zheng et al. recently introduced Directed Representation Optimization (DRO). This method first identifies a direction of refusal in the low-dimensional representation space of the LLM by fitting a linear regression to the empirical refusal rates of known prompts. It then tailors the optimization of the safety prompts to steer harmful queries towards this direction of refusal, while directing harmless queries in the opposite direction.</p>

                            ` },
                        { name: "Generation Intervention-based Defense", info: `
<p>Generation intervention-based defenses modify the original LLM response generation process to enhance safety. </p>

<p>Li et al. introduce the Rewindable Auto-regressive INference (RAIN) method. In this approach, the LLM tentatively produces tokens and evaluates their safety. If tokens are deemed safe, they are retained, and the generation process continues. If not, the model reverts to the beginning of these tokens and explores alternative tokens, ensuring only safe outputs proceed.</p>

<p>Xu et al. propose a new method, namely SafeDecoding, that fine-tunes a safety expert model derived from the original LLM using a curated safety dataset. During inference, this method adjusts the output probabilities of the original LLM by aligning them with the discrepancies observed between the safety expert model's and the original LLM's output distributions. This adjustment reduces the likelihood of generating unsafe outputs while increasing the probability of producing safe responses.</p>
                  
                            `},
                        { name: "Response Evaluation-based Defense", info: `
                           <p>Response evaluation-based defenses assess the harmfulness of LLM responses and often refine them afterward iteratively to make the responses safer. </p>

<p>Helbling et al. introduce a method where an auxiliary LLM evaluates the harmfulness of responses from the primary model to ensure safety. Going a step further, Pisano et al. employ a secondary LLM not only to assess harm but also to guide the refinement of responses. Similarly, Zeng et al. deploy several external LLMs serving different roles for assessing potential harm in responses and refining them. Instead of relying on additional LLMs, Kim et al. develop a methodology where the primary LLM itself evaluates and iteratively refines its outputs.</p>
 
                            ` },
                        { name: "Model Fine Tuning-based Defense", info: `
<p>Rather than relying on external measures, Model Fine-tuning-based defenses enhance LLM safety by altering the model's inherent characteristics. </p>

<p>This defense strategy encompasses a broad spectrum of techniques, each of which provides a unique perspective on enhancing model safety. Jain et al. and Bhardwaj and Poria train the model on a mixture of benign and adversarial data to enhance model safety, marking an initial step in this direction. Ge et al. introduce an adversarial framework designed for automatic red-teaming, which pits an attack model against the target LLM, with the former striving to refine attack prompts based on past successes, and the latter aiming to generate safe and helpful responses informed by previous interactions and feedback from a reward model.</p>

<p>Recognizing the inherent conflict between helpfulness and safety as a factor of irresponsible output, Zhang et al. fine-tune the LLM to explicitly recognize these objectives and prioritize safety during inference. In the fine-tuning stage, the LLM is exposed to both harmful and harmless prompts, accompanied by instructions that prioritize either helpfulness or safety. The LLM learns to produce helpful responses when prompted with helpfulness instructions and safe responses when prompted with safety instructions. During inference, the LLM is always presented with safety instructions alongside the prompt, guiding it to generate safe responses.</p>

<p>Drawing inspiration from the realm of backdoor attacks, Wang et al. tackle fine-tuning-based jailbreak attacks by embedding a secret prompt within safety examples during fine-tuning. Then during inference, the safety can be enhanced by prefixing these secret prompts to input prompts.</p>

<p>In a separate approach, Hasan et al. utilize model compression to enhance the LLM's resilience against jailbreaks. Their method employs a pruning technique known as Wanda, which has been shown to bolster the model's defenses.</p>

<p>Venturing into the innovative realm of knowledge editing as a new frontier for LLM post-training adjustments, Wang et al. introduce Detoxifying with Intraoperative Neural Monitoring (DINM), a technique that locates and modifies the weights of layers identified as sources of toxic outputs.</p>

<p>Taking a radical approach, Piet et al. propose abandoning the LLM's instruction tuning capability, instead training task-specific models from a non-instruction-tuned base model to sidestep potential attacks.</p>
                            ` }
                    ]
                }
            ]
        },
        {
            name: "Threats in VLMs",
            children: [
                {
                    name: "Jailbreaks on VLMs",
                    children: [
                        { name: "Prompt-to-image injection Jailbreak", info: `
                            <p>Recent studies have highlighted the vulnerability of VLMs to prompt-to-image injection attacks, which involve transferring harmful content into images with instructions. It includes paraphrasing a prohibited text query, converting it into a typographic visual prompt image, and using an incitement text prompt to motivate the model to answer the visual prompt. The process transforms the original query into a jailbreaking query that combines the visual prompt encoding the question and the incitement prompt to generate the final response.</p>

<p>Gong et al. proposed FigStep as a black-box approach for jailbreaking. It feeds harmful instructions into VLMs through the image channel and then uses benign text prompts to induce VLMs to output contents that violate common AI safety policies. They also found that the safety of VLMs requires attention beyond what is provided by LLMs, due to inherent limitations in text-centric safety alignment.</p>

<p>The approach used in prompt-to-image injection attacks shares similarities with the demonstration-based jailbreaks discussed in Demonstration Based Jailbreak section}. In both cases, the attacker crafts specific prompts (either text-based for LLMs or image-based for VLMs) to guide the model's response toward generating content that violates safety policies. However, prompt-to-image injection attacks exploit the additional attack surface introduced by the visual modality in VLMs, allowing adversaries to bypass the safety measures that primarily focus on textual inputs.</p>


                            ` },
                        { name: "Prompt-Image Perturbation Injection Jailbreak", info: `
 <h4>Prompt-Image Perturbation Injection Jailbreaks</h4>
<p>Prompt-Image Perturbation Injection Jailbreaks have been widely studied in VLMs, particularly in the black-box setting. It involves manipulating both the textual and visual inputs. Specifically, the original text is first maliciously perturbed to describe a different and potentially harmful scenario. The original image is then perturbed by adding imperceptible noise. The perturbed texts, combined with perturbed images, are fed into the VLM, leading to responses that should be refused.</p>

<p>Early works, such as Co-Attack, focused on exploiting cross-modal interactions by perturbing image and text modalities collectively. However, it suffered from limitations in its transferability to other VLMs. To address these limitations, Lu et al. proposed the Set-Level Guidance Attack (SGA), which leverages modality interactions and incorporates alignment-preserving augmentation with cross-modal guidance. Despite its advancements, SGA has limitations in adequately addressing the optimal matching of post-augmentation image examples with their corresponding texts. Building on SGA, Han et al. developed the OT-Attack, which incorporates the theory of optimal transport to analyze and map data-augmented image sets and text sets, ensuring a balanced match after augmentation.</p>

<p>Despite the advancements mentioned above, challenges remain in effectively modeling inter-modal correspondence and optimizing the transferability of adversarial examples across different VLMs. Further improvements in transferability and adversarial example generation were made by Niu et al. and Qi et al., who introduced the concept of an image Jailbreaking Prompt (imgJP) and visual adversarial examples that show strong data-universal and model-transferability properties. Their approach enables black-box jailbreaking of various VLMs and can be converted to achieve LLM jailbreaks by transforming an imgJP to a text Jailbreaking Prompt (txtJP).</p>

<p>Carlini et al. demonstrated that VLMs can be easily exploited by NLP-based optimization attacks, inducing them to perform arbitrary unaligned behavior through adversarial perturbation of the input image. The authors conjecture that improved NLP attacks may demonstrate a similar level of adversarial control over text-only models.</p>

<p>To further improve the transferability of adversarial examples, Luo et al. introduced the Cross-Prompt Attack (CroPA). These prompts are generated by optimizing in the opposite direction of the perturbation, thereby covering more prompt embedding space and significantly improving transferability across different prompts. Zhao et al. conducted a quantitative evaluation of the adversarial robustness of different VLMs by generating adversarial images that deceive the models into producing targeted responses. Similarly, Schlarmann \& Hein and Bailey et al. demonstrated the high attack success rate on VLMs by imperceptible perturbations. Along the line, Zhou et al. propose AdvCLIP for generating downstream-agnostic adversarial examples in multimodal contrastive learning. Yin et al. propose VLATTACK, which generates adversarial samples by fusing perturbations of images and texts from both single-modal and multimodal levels.</p>

<p>Generally speaking, the recent advancements in Image Perturbation Injection Jailbreaks on VLMs share several similarities with the gradient-based and evolutionary-based jailbreaks discussed in Gradient Based Jailbreak section and Evolutionary-Based Jailbreak section for LLMs. Both types of attacks leverage optimization techniques to generate adversarial inputs and further use iterative processes for effectiveness improvement. However, VLMs process both visual and textual inputs, and the interactions between these modalities can be exploited by attackers. Image Perturbation Injection attacks specifically target these cross-modal interactions to generate more potent and harder-to-detect adversarial examples. Moreover, the inclusion of visual inputs in VLMs expands the attack surface compared to text-only LLMs. Attackers can manipulate both the textual and visual components of the input, which allows for the jailbreaking of VLMs and LLMs using a single adversarial image. Additionally, optimizing the transferability of adversarial examples across different VLMs is a significant challenge, as the cross-modal interactions and architectures of these models can vary greatly. This encourages the development of more generalizable adversarial perturbations.</p>

<p>The advancements made in each study, from early approaches like Co-Attack and Sep-Attack to more sophisticated methods like SGA, OT-Attack, and jailbreaking attacks, have progressively expanded our understanding of the adversarial landscape in VLMs. However, the challenges in effectively modeling inter-modal correspondence, optimizing transferability across different VLP models, and defending against emerging attacks like image hijacks and AdvCLIP underscore the importance of continued research efforts in this field.</p>
                            

                            ` },
                        { name: "Proxy Model Transfer Jailbreak", info: `
                           <p>Proxy Model Transfer Jailbreaks leverage the transferability of malicious manipulation to conduct attacks. Attackers may use proxy models to create adversarial examples that are more likely to transfer to the victim model. Similar to rule-based jailbreaks discussed in Rule Based Jailbreak section for LLMs, attackers do not have direct access to the model's parameters or architecture. Attackers have white-box access to a proxy model, which is used to generate adversarial examples. They apply various transferability-enhancing techniques, such as input diversity, momentum, or translation-invariant attacks, to create adversarial examples that are more likely to transfer to the victim model. The crafted adversarial examples are then transferred to the black-box victim model. If the transfer is successful, the victim model misclassifies the adversarial examples, leading to a jailbroken output.</p>

<p>Proxy Model Transfer Jailbreaks exploit the transferability of adversarial examples across different models. This approach builds upon the foundational work by Liu et al., Papernot et al., Xie et al., Dong et al., and Yang et al.</p>

<p>Most recently, the exploration into adversarial robustness by Dong et al. revealed vulnerabilities specific to commercial VLMs, proposing novel attacks tailored to the multimodal processing of these models. Chen et al. introduce a novel perspective on model ensembles in Proxy Model Transfer Jailbreaks. They define the common weakness of model ensembles as a solution that lies in a flat loss landscape and is close to the local optima of each model. By promoting these two properties, the authors aim to generate more transferable adversarial examples that can effectively fool black-box models like Google’s Bard.</p>

<p>Shayegani et al. introduce a novel perspective on the vulnerability of multi-modal systems that incorporate off-the-shelf components, such as pre-trained vision encoders like CLIP, in a plug-and-play manner. They propose adversarial embedding space attacks, which exploit the vast and under-explored embedding space of these pre-trained encoders without requiring access to the multi-modal system's weights or parameters. However, instead of using a substitute VLM to generate perturbed images, as in the works of Liu et al. and Papernot et al., the proposed method directly exploits the embedding space of the vision encoder, making it more efficient and potentially more effective.</p>

<p>Despite the advancements in Proxy Model Transfer Jailbreaks, some limitations and challenges need to be addressed. As highlighted by Zhao et al., Proxy model transfer attacks depend on having white-box access to proxy models. This necessity can limit the applicability of these attacks in situations where similar models are not available or accessible. Generating adversarial examples, especially high-quality ones that are likely to transfer, can be computationally expensive. This process involves finding input perturbations that lead to misclassifications, which may require significant computational resources, especially for complex VLMs. Besides, the success of Proxy Model Transfer Jailbreaks heavily relies on the similarity between the victim and proxy models. Differences in architectures, training data, or optimization objectives can reduce the transferability of adversarial examples. This implies a potential limitation in the universality of these attacks across diverse models.</p>
                            ` }
                    ]
                },
                {
                    name: "Defense on VLMs",
                    children: [
                        { name: "Model Fine Tuning-based Defense", info: `
<p>Model fine-tuning-based defenses focus on intercepting and mitigating jailbreak prompts during the model's training phase, leveraging techniques such as prompt optimization and natural language feedback to reinforce the model's resistance to malicious inputs. These methods mitigate the risks associated with malicious inputs, particularly Prompt-to-image Injection Jailbreaks, which exploit the multimodal capabilities of VLMs to bypass safety mechanisms.</p>

<p>In addressing the challenges of ensuring the safety of VLMs without compromising their performance, Chen et al. first introduced DRESS, which focuses on leveraging natural language feedback from large language models to improve the alignment and interactions within VLMs. By categorizing NLF into critique and refinement feedback, DRESS aims to enhance the model's ability to generate more aligned and helpful responses, as well as to refine its outputs based on the feedback received. This approach addresses the limitations of prior VLMs that rely solely on supervised fine-tuning and RLHF for alignment with human preferences. DRESS introduces a method for conditioning the training of VLMs on both critique and refinement NLF, thus fostering better multi-turn interactions and alignment with human values.</p>

<p>In the follow-up, Wang et al. propose Adashield, a prompt-based defense mechanism that does not necessitate fine-tuning of VLMs or the development of auxiliary models. This approach is particularly advantageous as it leverages a limited number of malicious queries to optimize defense prompts, thereby circumventing the challenges associated with high computational costs, significant inference time costs, and the need for extensive training data. Through an auto-refinement framework that includes a target VLM and a defender LLM, AdaShield iteratively optimizes defense prompts. This process generates a diverse pool of defense prompts that adhere to specific safety guidelines, enhancing the robustness of VLMs against Prompt-to-image Injection Jailbreak. The adaptive and automatic nature of this approach ensures that VLMs are safeguarded effectively without requiring extensive modifications to the models themselves. Comparatively, Pi et al. represents a more traditional approach for defense, incorporating a harm detector and a detoxifier to correct potentially harmful outputs generated by VLMs. However, this Model Fine-tuning-based Defense strategy requires a significant amount of high-quality data and computational resources. Additionally, as a post-hoc filtering defense mechanism, it incurs substantial inference time costs, which can be a significant drawback in practical applications.</p>

<p>The Model Fine-tuning-based Defense strategies share similarities with the model fine-tuning defenses discussed in Defense on LLM section for LLMs. Both approaches aim to enhance the model's safety and alignment with human preferences by modifying the training process. However, the multi-modal nature of VLMs introduces additional challenges and opportunities for Model Fine-tuning-based Defense strategies. The presence of both textual and visual modalities in VLMs necessitates the development of defense mechanisms that can effectively align these modalities in a compositional manner. For instance, AdaShield addresses this challenge by generating defense prompts that adhere to specific safety guidelines. DRESS, on the other hand, leverages natural language feedback to improve the alignment and interaction between the textual and visual modalities in VLMs.</p>

                            ` },
                        { name: "Response Evaluation-based Defense", info: `
<p>Response evaluation-based defenses operate during the model's inference phase, ensuring that the model's response to a jailbreak prompt remains safe and aligned with the desired behavior.</p>

<p>Pi et al. and Zong et al. proposed Model Fine-tuning-based Defense methods that aim to align VLMs with specially constructed red-teaming data. However, these approaches are labor-intensive and may not cover all potential attack vectors. On the other hand, inference-based defense methods focus on protecting VLMs during the inference stage without requiring additional training.</p>

<p>One notable Response Evaluation-based approach is ECSO, a training-free method that exploits the inherent safety awareness of VLMs. ECSO leverages the observation that VLMs can detect unsafe content in their own responses and that the safety mechanism of pre-aligned LLMs persists in VLMs but is suppressed by image features. By transforming potentially malicious visual content into plain text using a query-aware image-to-text transformation, ECSO effectively restores the intrinsic safety mechanism of the pre-aligned LLMs within the VLM.</p>

<p>The response evaluation-based defense strategies discussed in this section underscore the importance of developing defense mechanisms that can effectively detect and mitigate potentially harmful content generated by VLMs during the inference stage. While some of the response evaluation defenses developed for LLMs  may be adapted to the VLM context, dedicated research efforts are required to address the unique challenges posed by the multi-modal nature of these models. By focusing on compositional safety alignment approaches and exploiting the inherent safety awareness of VLMs, Response Evaluation-based Defense strategies can provide an additional layer of protection against adversarial attacks on VLMs, complementing the Model Fine-tuning-based Defense strategies discussed in the previous section.</p>
                            ` },
                        { name: "Prompt Perturbation-based Defense", info: ` 
<p>Prompt perturbation-based Defense takes a different approach by altering the input prompts into mutated queries and analyzing the consistency of the model's responses to identify potential jailbreaking attempts, exploiting the inherent fragility of attack queries. </p>

<p>The prompt perturbation methods exploit the inherent fragility of attack queries, which often rely on crafted templates or complex perturbations, making them less robust than benign queries. By mutating the input into variant queries and analyzing the consistency of the language model's responses, prompt perturbation-based methods can effectively identify potential jailbreaking attempts. Zhang et al. proposed JailGuard, a prompt perturbation-based jailbreaking detection framework that supports both image and text modalities. JailGuard employs a variant generator with 19 mutators, including random and advanced mutators, to disturb the input queries and generate variants. The attack detector then analyzes the semantic similarity and divergence between the responses to these variants, identifying potential attacks when the divergence exceeds a predefined threshold. Evaluations on a multi-modal jailbreaking attack dataset demonstrate JailGuard's effectiveness, outperforming state-of-the-art defense methods.</p>

<p>The prompt perturbation-based defense strategies share similarities with those in Perturbation-Based Defense section for LLMs. Both approaches aim to detect and mitigate potentially harmful content by manipulating the input prompts. In the case of LLMs, prompt perturbation-based defenses involve perturbing the input prompts to generate multiple variations and then aggregating the model's responses to these variations to dilute the impact of adversarial prompts. The multi-modal nature of VLMs introduces additional challenges for prompt perturbation-based defense strategies. VLMs require defense mechanisms that can manipulate and analyze both textual and visual inputs. Prompt perturbation-based defense techniques like JailGuard provide an additional defense layer against jailbreaking attacks on VLMs without relying heavily on domain-specific knowledge or post-query analysis. These strategies complement model fine-tuning-based defenses and response evaluation-based defenses approaches, offering a comprehensive framework for safeguarding VLMs against adversarial attacks.</p>
                            ` }
                    ]
                }

            ]
        }
    ]
};

const margin = { top: 20, right: 90, bottom: 30, left: 90 };
const width = 1200 - margin.left - margin.right;
const height = 700 - margin.top - margin.bottom;

const svg = d3.select("#tree-container").append("svg")
    .attr("width", width + margin.right + margin.left)
    .attr("height", height + margin.top + margin.bottom)
    .append("g")
    .attr("transform", "translate(" + margin.left + "," + margin.top + ")");

const duration = 750;
const root = d3.hierarchy(treeData, d => d.children);
root.x0 = height / 2;
root.y0 = 0;
const tree = d3.tree().size([height, width]);

let i = 0;

update(root);

function update(source) {
    const treeData = tree(root);
    const nodes = treeData.descendants();
    const links = treeData.descendants().slice(1);

    nodes.forEach(d => { d.y = d.depth * 220; });

    const node = svg.selectAll('g.node')
        .data(nodes, d => d.id || (d.id = ++i));

    const nodeEnter = node.enter().append('g')
        .attr('class', 'node')
        .attr("transform", d => `translate(${source.y0},${source.x0})`)
        .on('click', (event, d) => {
            if (d.children) {
                d._children = d.children;
                d.children = null;
            } else {
                d.children = d._children;
                d._children = null;
            }
            update(d);
            if (d.data.info) {
                const modal = new bootstrap.Modal(document.getElementById('infoModal'));
                document.getElementById('infoModalLabel').textContent = d.data.name;
                document.querySelector('#infoModal .modal-body').innerHTML = d.data.info;
                MathJax.typeset();
                modal.show();
            }
        });;

    nodeEnter.append('circle')
        .attr('r', 1e-6)
        .style("fill", d => d._children ? "#a91d3a" : "#eeeeee");

    nodeEnter.append('text')
        .attr("dy", ".35em")
        .attr("x", d => d.children || d._children ? -13 : 13)
        .attr("text-anchor", d => d.children || d._children ? "end" : "start")
        .text(d => d.data.name)
        .style("font-size", '14px')
        .style("fill-opacity", 1e-6)
        .attr("class", "node-label");

    const nodeUpdate = nodeEnter.merge(node);

    nodeUpdate.transition()
        .duration(duration)
        .attr("transform", d => `translate(${d.y},${d.x})`)
        .on('end', function () {
            nodeUpdate.selectAll('circle, text')
                .on('mouseover', function (event, d) {
                    d3.select(this.parentNode).select('circle').transition().duration(200).attr('r', 12).style('fill', '#a91d3a');
                    d3.select(this.parentNode).select('text').transition().duration(200).style('font-size', '16px');
                })
                .on('mouseout', function (event, d) {
                    d3.select(this.parentNode).select('circle').transition().duration(200).attr('r', 10).style('fill', d => d._children ? "#c73659" : "#eeeeee");
                    d3.select(this.parentNode).select('text').transition().duration(200).style('font-size', '14px');
                })
        });

    nodeUpdate.select('circle')
        .attr('r', 10)
        .style("fill", d => d._children ? "#c73659" : "#eeeeee")
        .attr('cursor', 'pointer');

    nodeUpdate.select('text')
        .style("fill-opacity", 1);

    const nodeExit = node.exit().transition()
        .duration(duration)
        .attr("transform", d => `translate(${source.y},${source.x})`)
        .remove();

    nodeExit.select('circle')
        .attr('r', 1e-6);

    nodeExit.select('text')
        .style('fill-opacity', 1e-6);

    const link = svg.selectAll('path.link')
        .data(links, d => d.id);

    const linkEnter = link.enter().insert('path', "g")
        .attr("class", "link")
        .attr('d', d => {
            const o = { x: source.x0, y: source.y0 };
            return diagonal(o, o);
        });

    const linkUpdate = linkEnter.merge(link);

    linkUpdate.transition()
        .duration(duration)
        .attr('d', d => diagonal(d, d.parent));

    link.exit().transition()
        .duration(duration)
        .attr('d', d => {
            const o = { x: source.x, y: source.y };
            return diagonal(o, o);
        })
        .remove();

    nodes.forEach(d => {
        d.x0 = d.x;
        d.y0 = d.y;
    });

    function diagonal(s, d) {
        return `M ${s.y} ${s.x}
            C ${(s.y + d.y) / 2} ${s.x},
                ${(s.y + d.y) / 2} ${d.x},
                ${d.y} ${d.x}`;
    }
}

anime({
    targets: '#tree-container',
    opacity: [0, 1],
    translateY: [50, 0],
    easing: 'easeOutExpo',
    duration: 1500
});
