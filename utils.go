package mlp

import (
	"encoding/csv"
	"errors"
	"io"
	"log"
	"math"
	"os"
	"strconv"
)

func sigmoid(x float64) float64 {
	return 1 / (1 + math.Exp(-x))
}

func dsigmoid(x float64) float64 {
	return x * (1 - x)
}

func arrayStringToInt(s []string) ([]float64, error) {
	ints := make([]float64, len(s))
	for i := range s {
		v, err := strconv.ParseFloat(s[i], 64)
		if err != nil {
			return nil, errors.New("No se pudo convertir el arreglo de string en enteros")
		}
		ints[i] = v
	}
	return ints, nil
}

//ReadData read a file and return a matrix for use to train o target data
func ReadData(fileName string) [][]float64 {
	t, err := os.Open(fileName)
	if err != nil {
		log.Fatal(err.Error())
	}
	data := make([][]float64, 0)
	r := csv.NewReader((t))
	r.Comment = '#'
	for {
		record, err := r.Read()
		if err == io.EOF {
			break
		}
		if err != nil {
			log.Fatal(err)

		}
		d, err := arrayStringToInt(record)
		if err != nil {
			log.Fatal(err)

		}
		data = append(data, d)
	}
	return data
}
