**1. Deep Learning cheatsheet**

&#10230; 1. Deep Learning cheatsheet

<br>


**2. Neural Networkd**

&#10230; 2. Reti Neurali

<br>


**3. Neural networks are a class of models that are built with layers. Commonly used types of neural networks include convolutional and recurrent neural networks.**

&#10230; 3. Le reti neurali (o neural netowrk) sono una classe di modelli stratificati (basati su diversi layer). I tipi di neural network principalmente utilizzati sono le reti convoluzionali o ricorrenti.

<br>

**4. Architecture ― The vocabulary around neural networks architectures is described in the figure below:**

&#10230; 4. Architettura - Il vocabolario tipico delle architetture delle reti neurali è descritto nella figura sottostante.

<br>

**5. [Input layer, hidden layer, output layer]**

&#10230; 5. [Layer di input (input layer), layer nascosti (hidden layer), layer di output (output layer)]

<br>

**6. By noting i the ith layer of the network and j the jth hidden unit of the layer, we have:**

&#10230; 6. Indicando con i, l'iesimo layer della rete e con j, la j-esima unità nascosta del livello, abbiamo:
<br>

**7. where we note w, b, z the weight, bias and output respectively.**

&#10230; 7. dove w, b, z sono rispettivamente i pesi, il bias e l'output.

<br>

**8. Activation function ― Activation functions are used at the end of a hidden unit to introduce non-linear complexities to the model. Here are the most common ones:**

&#10230; 8. Funzione di attivazione - Le funzioni di attivazione sono utilizzate alla fine di un hidden layer, per introdurre una complessità non lineare nel modello. Le più comuni sono: 

<br>

**9. [Sigmoid, Tanh, ReLU, Leaky ReLU]**

&#10230;  9. [Sigmoide, Tangente iperbolica (Tanh), ReLU (unità lineare rettificata), Leaky ReLU (unità lineare rettificata di tipo leaky) ]

<br>

**10. Cross-entropy loss ― In the context of neural networks, the cross-entropy loss L(z,y) is commonly used and is defined as follows:**

&#10230; 10. Cross-entropy loss (Perdita di entropia incrociata) - Nel contesto delle reti neurali la perdita di entropia incrociata è comunemente utilizzata e definita come segue:

<br>

**11. Learning rate ― The learning rate, often noted α or sometimes η, indicates at which pace the weights get updated. This can be fixed or adaptively changed. The current most popular method is called Adam, which is a method that adapts the learning rate.**

&#10230; 11. Velocità di apprendimento (Learning rate) - Il learning rate, spesso indicato come α oppure η, indica la velocità con cui i pesi vengono modificati. Può essere fisso o adattarsi dinamicamente. Il metodo più popolare al momento è Adam ed è un metodo capace di modificare il learning rate.

<br>

**12. Backpropagation ― Backpropagation is a method to update the weights in the neural network by taking into account the actual output and the desired output. The derivative with respect to weight w is computed using chain rule and is of the following form:**

&#10230; 12. Backpropagation - La Backpropagation è un metodo di aggiornamento dei pesi della rete neurale, basato sul confronto degli output realmente ottenuti e di quelli desiderato. La derivata rispetto al peso w è calcolata utilizzando una catena di regole e ha forma la seguente:

<br>

**13. As a result, the weight is updated as follows:**

&#10230; 13. Di conseguenza, il peso è aggiornato nel modo seguente:

<br>

**14. Updating weights ― In a neural network, weights are updated as follows:**

&#10230; 14. Aggiornamento dei pesi - In una rete neurale i pesi sono aggiornati come segue:

<br>

**15. Step 1: Take a batch of training data.**

&#10230; 15. Step 1: Si prende un sottoinsieme dei dati di training.

<br>

**16. Step 2: Perform forward propagation to obtain the corresponding loss.**

&#10230; 16. Step 2: Si effettua una forward propagation per ottenere la perdita corrispondente.

<br>

**17. Step 3: Backpropagate the loss to get the gradients.**

&#10230; 17. Step 3: Si propaga indietro la perdita per ottenere i gradienti.

<br>

**18. Step 4: Use the gradients to update the weights of the network.**

&#10230; Step 4: Si usano i gradienti per aggiornare i pesi della rete.

<br>

**19. Dropout ― Dropout is a technique meant at preventing overfitting the training data by dropping out units in a neural network. In practice, neurons are either dropped with probability p or kept with probability 1−p**

&#10230; 19: Dropout - Il dropout è una tecnica volta a prevenire il fenomeno dell'overfitting sui dati di training escludendo dei neuroni della rete neurale. I neuroni vengono esclusi (spenti) con probabilità p o tenuti con probabilità 1-p.

<br>

**20. Convolutional Neural Networks**

&#10230; 20. Reti neurali convoluzionali (Convolutional Neural Network).

<br>

**21. Convolutional layer requirement ― By noting W the input volume size, F the size of the convolutional layer neurons, P the amount of zero padding, then the number of neurons N that fit in a given volume is such that:**

&#10230; 21. Requisiti di un layer convoluzionale - Indicando con W la dimensione del volume di input, con F la dimensione del layer di neuroni convoluzionali, con P il numero di padding zero, allora il numero di neurono N the fitta il un dato volume è pari a:

<br>

**22. Batch normalization ― It is a step of hyperparameter γ,β that normalizes the batch {xi}. By noting μB,σ2B the mean and variance of that we want to correct to the batch, it is done as follows:**

&#10230; 22. Batch normalization -  è uno step di iper-parametri γ,β che normalizza il batch {xi}. Indicando con μB,σ2B rispettivamente la media e la varianza di quanto vogliamo normalizzare il batch, allora:

<br>

**23. It is usually done after a fully connected/convolutional layer and before a non-linearity layer and aims at allowing higher learning rates and reducing the strong dependence on initialization.**

&#10230; 23. Tipicamente è applicata dopo un layer totalmente connesso o convoluzionale e prima di un layer non lineare al fine di consentire learning rate più alti e ridurre dipendenze forti dovute all'inizializzazione.

<br>

**24. Recurrent Neural Networks**

&#10230; 24. Reti neurali ricorrenti (RNN)

<br>

**25. Types of gates ― Here are the different types of gates that we encounter in a typical recurrent neural network:**

&#10230; 25. Tipi di gate - A seguire le differenti tipologie di gate che è possibile incontrare in una tipica rete ricorrente.

<br>

**26. [Input gate, forget gate, gate, output gate]**

&#10230; 26. [Input gate, forget gate, gate, output gate]

<br>

**27. [Write to cell or not?, Erase a cell or not?, How much to write to cell?, How much to reveal cell?]**

&#10230; 27. Scrivere una cella o no? Svuotare una cella o no, Quanto scrivere? Quanto rivelare?

<br>

**28. LSTM ― A long short-term memory (LSTM) network is a type of RNN model that avoids the vanishing gradient problem by adding 'forget' gates.**

&#10230; 28. LSTM - Una rete long short-term memory (LSTM) è un tipo di RNN che evita il problema della scomparsa del gradiente aggiungendo dei 'forget' gate.

<br>

**29. Reinforcement Learning and Control**

&#10230; 29. Reinforcement Learning e Controllo
<br>

**30. The goal of reinforcement learning is for an agent to learn how to evolve in an environment.**

&#10230; 30. Il reinforcement learning ha il fine di rendere un agente capace di apprendere come evolvere in un ambiente

<br>

**31. Definitions**

&#10230; 31. Definizioni

<br>

**32. Markov decision processes ― A Markov decision process (MDP) is a 5-tuple (S,A,{Psa},γ,R) where:**

&#10230; 32. Processo decisionale di Markov - Un processo decisionale di Markov (MDP) è una 5-tupla (S,A,{Psa},γ,R) dove:

<br>

**33. S is the set of states**

&#10230; 33. S è l'insieme degli stati

<br>

**34. A is the set of actions**

&#10230; 34. A è l'insieme delle azioni

<br>

**35. {Psa} are the state transition probabilities for s∈S and a∈A**

&#10230; 35. {Psa} sono le probabilità di transizione di stato per s∈S e a∈A

<br>

**36. γ∈[0,1[ is the discount factor**

&#10230; 36. γ∈[0,1[ è il fattore di sconto

<br>

**37. R:S×A⟶R or R:S⟶R is the reward function that the algorithm wants to maximize**

&#10230; 37. R:S×A⟶R o R:S⟶R è la funzione premio (reward) che l'algoritmo deve massimizzare

<br>

**38. Policy ― A policy π is a function π:S⟶A that maps states to actions.**

&#10230;

<br>

**39. Remark: we say that we execute a given policy π if given a state a we take the action a=π(s).**

&#10230;

<br>

**40. Value function ― For a given policy π and a given state s, we define the value function Vπ as follows:**

&#10230;

<br>

**41. Bellman equation ― The optimal Bellman equations characterizes the value function Vπ∗ of the optimal policy π∗:**

&#10230;

<br>

**42. Remark: we note that the optimal policy π∗ for a given state s is such that:**

&#10230;

<br>

**43. Value iteration algorithm ― The value iteration algorithm is in two steps:**

&#10230;

<br>

**44. 1) We initialize the value:**

&#10230;

<br>

**45. 2) We iterate the value based on the values before:**

&#10230;

<br>

**46. Maximum likelihood estimate ― The maximum likelihood estimates for the state transition probabilities are as follows:**

&#10230;

<br>

**47. times took action a in state s and got to s′**

&#10230;

<br>

**48. times took action a in state s**

&#10230;

<br>

**49. Q-learning ― Q-learning is a model-free estimation of Q, which is done as follows:**

&#10230;
