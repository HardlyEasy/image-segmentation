import os
import time

from src.Model import Model
from src.View import View
from src.Controller import MatrixController, EdgeController, \
    SegmentController, OutputController


def main():
    print('Starting ... be patient with many or large images')
    start_time = time.time()
    model = Model()
    view = View()

    matrix_controller = MatrixController(model, view)
    edge_controller = EdgeController(model, view)
    segment_controller = SegmentController(model, view)
    output_controller = OutputController(model, view)

    matrix_controller.run()
    edge_controller.run()
    segment_controller.run()
    output_controller.run()

    end_time = time.time()
    print(round(end_time - start_time, 2), 'seconds total runtime')
    print("End of program.")


if __name__ == "__main__":
    main()
