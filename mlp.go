package mlp

import (
	"errors"
	"math/rand"
	"time"
)

//MultiLayerPerceptron struct for the multilayer perceptron
type MultiLayerPerceptron struct {
	Input        []float64
	Weigths      []Matrix
	Hidden       []Matrix
	Output       []float64
	LearningRate float64
}

//NewMultiLayerPerceptron constructor for a new multilayer perceptron
func NewMultiLayerPerceptron(inputs, hiddens, outputs int, lr float64) *MultiLayerPerceptron {
	Inputs := make([]float64, inputs+1)
	Inputs[inputs] = 1
	Outputs := make([]float64, outputs)
	Hiddens := make([]Matrix, 1)
	Hiddens[0] = NewMatrix(hiddens+1, 1)
	Hiddens[0][hiddens][0] = 1
	Weigths := make([]Matrix, 2)
	Weigths[0] = NewRandomizeMatrix(len(Hiddens[0]), len(Inputs), -1, 1)
	Weigths[1] = NewRandomizeMatrix(len(Outputs), len(Hiddens[0]), -1, 1)
	ml := &MultiLayerPerceptron{
		Input:        Inputs,
		Hidden:       Hiddens,
		Output:       Outputs,
		Weigths:      Weigths,
		LearningRate: lr,
	}
	return ml
}

//AddHiddenLayer add a hidden layer into mlp
func (ml *MultiLayerPerceptron) AddHiddenLayer(neurons int) {
	// aux := ml.Weigths[len(ml.Weigths)]
	ml.Weigths[len(ml.Weigths)-1] = NewRandomizeMatrix(neurons+1,
		len(ml.Hidden[len(ml.Hidden)-1]), -1, 1)
	ml.Weigths = append(ml.Weigths, NewRandomizeMatrix(len(ml.Output),
		neurons+1, -1, 1))
	ml.Hidden = append(ml.Hidden, NewRandomizeMatrix(neurons+1, 1, -1, 1))
	ml.Hidden[0][len(ml.Hidden)-1][0] = 1
}

func (ml *MultiLayerPerceptron) setInput(input []float64) error {
	if len(input) != len(ml.Input)-1 {
		return errors.New("la entrada no coincide con la dada inicialmente")
	}
	copy(ml.Input[:len(input)], input)
	return nil
}

//feedForwardPropagation feedfordward propagation algorithm
func (ml *MultiLayerPerceptron) feedForwardPropagation() {
	var err error
	//calcula la salida de las neuronas ocultas
	ml.Hidden[0], err = MatrixMultiplication(ml.Weigths[0],
		Transpose([][]float64{ml.Input}))

	if err != nil {
		panic(err.Error())
	}
	//aplica la funcion sigmoide
	ml.Hidden[0] = Map(ml.Hidden[0], sigmoid)

	//se calcula las salidas de todas las capas ocultas despues de la primera
	for i := 1; i < len(ml.Hidden); i++ {
		var err error
		ml.Hidden[i], err = MatrixMultiplication(ml.Weigths[i], ml.Hidden[i-1])
		if err != nil {
			panic(err.Error())
		}
		ml.Hidden[i] = Map(ml.Hidden[i], sigmoid)

	}
	//se calcula la salida
	aux, err := MatrixMultiplication(ml.Weigths[len(ml.Hidden)],
		ml.Hidden[len(ml.Hidden)-1])
	if err != nil {
		panic(err.Error())
	}
	ml.Output = Transpose(Map(aux, sigmoid))[0]

}

//backFordwardPropagation back forward propagation algorithm
func (ml *MultiLayerPerceptron) backFordwardPropagation(deltaE []float64) Matrix {
	//calculo de los deltas para el backpropagation
	//se multiplica los deltas de error de la salida por la tasa de aprendizaje
	outputG := MatrixScalar(([][]float64{deltaE}), ml.LearningRate)
	//se multiplica el anterior resultado por las neuronas de salida aplicandoles la derivada de la sigmoide
	outputG, err := MatrixWiseMultiplication(outputG,
		(Map([][]float64{ml.Output}, dsigmoid)))
	if err != nil {
		panic(err.Error())
	}
	//se multiplica por la salida de las neuronas ocultas de la ultima capa por el resultado anterior
	deltas, err := MatrixMultiplication(Transpose(outputG),
		Transpose(ml.Hidden[len(ml.Hidden)-1]))
	if err != nil {
		panic(err.Error())
	}
	//se calculan los delta de error de las neuronas ocultas de la ultima capa
	hError, err := MatrixMultiplication(Transpose(ml.Weigths[len(ml.Hidden)]),
		Transpose([][]float64{deltaE}))
	if err != nil {
		panic(err.Error())
	}

	//se ajustan los pesos con los deltas de pesos calculados
	if err := ml.Weigths[len(ml.Hidden)].Add(deltas); err != nil {
		panic(err.Error())
	}

	//se calculan los deltas de todas las capas ocultas hasta la primera sin incluirla
	for i := len(ml.Hidden); i > 1; i-- {
		// println("back for")
		newOutputG := MatrixScalar(hError, ml.LearningRate)
		newOutputG, err = MatrixWiseMultiplication(newOutputG,
			(Map(ml.Hidden[i-1], dsigmoid)))
		if err != nil {
			panic(err.Error())
		}
		newDeltas, err := MatrixMultiplication(newOutputG,
			Transpose(ml.Hidden[i-2]))
		if err != nil {
			panic(err.Error())
		}
		// println(ml.Weigths[i-1])
		hError, err = MatrixMultiplication(Transpose(ml.Weigths[i-1]),
			hError)
		if err != nil {
			panic(err.Error())
		}
		if err := ml.Weigths[i-1].Add(newDeltas); err != nil {
			panic(err.Error())
		}

	}
	//se calculan los deltas de pesos de la entrada
	hiddenG := MatrixScalar(hError, ml.LearningRate)
	hiddenG, err = MatrixWiseMultiplication(hiddenG,
		(Map(ml.Hidden[0], dsigmoid)))
	if err != nil {
		panic(err.Error())
	}
	hDeltas, err := MatrixMultiplication(hiddenG,
		[][]float64{ml.Input})
	if err != nil {
		panic(err.Error())
	}
	if err := ml.Weigths[0].Add(hDeltas); err != nil {
		panic(err.Error())
	}

	return hDeltas

}

//Train funcion to train the mlp, given inputs and targets data
func (ml *MultiLayerPerceptron) Train(inputs, targets [][]float64, epochs int, verbose bool) {
	rand.Seed(time.Now().UnixNano())
	for i := 0; i < epochs; i++ {

		for _, index := range rand.Perm(len(inputs)) {
			input := inputs[index]
			if err := ml.setInput(input); err != nil {
				panic(err.Error())
			}
			ml.feedForwardPropagation()
			delta := make([]float64, len(targets[index]))
			for j := range ml.Output {
				delta[j] = targets[index][j] - ml.Output[j]

			}
			ml.backFordwardPropagation(delta)

			if verbose {
				fmt.Printf("epoch %d: deltas: %v", i, ml.backForwardPropagation(delta))
			}

		}

		// for i, input := range intputs {
		// 	if err := ml.setInput(input); err != nil {
		// 		panic(err.Error())
		// 	}
		// 	ml.feedForwardPropagation()
		// 	delta := make([]float64, len(targets[0]))
		// 	for j := range ml.Output {
		// 		delta[j] = targets[i][j] - ml.Output[j]

		// 	}
		// 	ml.backFordwardPropagation(delta)
		// }
	}
}

//Guess return the guess given a new input after training
func (ml *MultiLayerPerceptron) Guess(input []float64) []float64 {

	ml.setInput(input)
	ml.feedForwardPropagation()
	return ml.Output
}
