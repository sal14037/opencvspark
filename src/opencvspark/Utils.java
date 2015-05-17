package opencvspark;

import hipi.image.FloatImage;

import org.apache.log4j.Logger;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.MatOfByte;
import org.opencv.core.Rect;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;

public class Utils {

	public static Logger l = Logger.getLogger(Test.class);

	public static Mat floatImgToMat(FloatImage value) {
		float[] f = value.getData();
		int w = value.getWidth();
		int h = value.getHeight();
		int b = value.getBands();

		Mat m = new Mat(h, w, CvType.CV_32FC3);

		// Traverse image pixel data in raster-scan order and update running
		for (int j = 0; j < h; j++) {
			for (int i = 0; i < w; i++) {
				float[] fv = { f[(j * w + i) * 3 + 2] * 255,
						f[(j * w + i) * 3 + 1] * 255,
						f[(j * w + i) * 3 + 0] * 255 };
				m.put(j, i, fv);
			}
		}

		return m;
	}

	public static byte[] matToByteArr(Mat m) {
		MatOfByte buf = new MatOfByte();
		Imgcodecs.imencode(".jpg", m, buf);
		m.convertTo(m, CvType.CV_8UC1);
		byte[] b = buf.toArray();
		return b;
	}

	public static Mat cropMatrix(Mat input, Rect rect) {
		Mat output = new Mat();
		// check if the position is within the image
		// area
		if (rect.x >= 0 && rect.y >= 0 && rect.x <= input.cols()
				&& rect.y <= input.rows()) {
			if (rect.x + rect.width <= input.cols()
					&& rect.y + rect.height <= input.rows()) {
				// cropping within boundaries
				l.info("Person cropping within limits");
				output = input.submat(rect.y, rect.y + rect.height, rect.x,
						rect.x + rect.width);
			} else {
				// cropping outside of boundaries
				l.info("Person cropping outside limits");
				output = input.submat(rect.x, input.rows(), rect.y,
						input.cols());
			}
		}
		return output;
	}
}
