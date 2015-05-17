package opencvspark;
import org.opencv.core.Mat;
import org.opencv.core.MatOfRect;
import org.opencv.core.Scalar;
import org.opencv.imgproc.Imgproc;
import org.opencv.objdetect.CascadeClassifier;

public class FaceDetection {

	final Scalar rectColor = new Scalar(0, 0, 255);

	MatOfRect faces = new MatOfRect();
	CascadeClassifier face_cascade = new CascadeClassifier(
			"/home/thomas/opencv/opencv3/data/haarcascades/haarcascade_frontalface_default.xml");

	// /home/thomas/opencv/opencv3/data/haarcascades/haarcascade_frontalface_alt.xml

	public Mat detectFace(Mat m) {
		face_cascade.detectMultiScale(m, faces);
		return m;
	}

	public Mat enhance(Mat m) {
		// enhance image by combining grey and colour
		Mat mGrey = new Mat();
		m.copyTo(mGrey);
		Imgproc.cvtColor(m, mGrey, Imgproc.COLOR_BGR2GRAY);
		Imgproc.equalizeHist(mGrey, mGrey);
		return mGrey;
	}

}
