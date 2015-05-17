package opencvspark;
import org.opencv.core.Mat;
import org.opencv.core.Scalar;
import org.opencv.face.Face;
import org.opencv.face.LBPHFaceRecognizer;

public class FaceRecognition {

	private LBPHFaceRecognizer bf = Face.createLBPHFaceRecognizer();
	private int[] predicted = new int[1];
	private double[] confidence = new double[1];
	private final Scalar colour = new Scalar(255, 0, 0);
	private String savedFR = "/home/thomas/workspace/opencvcloud/test.yml";

	// FACERECOGNITION

	public void loadRecogniser() {
		bf.load(savedFR);
	}

	public int recognise(Mat m) {
		bf.predict(m, predicted, confidence);
		return 0;
	}

	public int getLabel() {
		return predicted[0];
	}

	public double[] getConfidence() {
		return confidence;
	}

	public Scalar getColour() {
		return colour;
	}

}
