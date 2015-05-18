package opencvspark;

import java.util.Arrays;
import java.util.List;

import hipi.image.FloatImage;

import org.apache.hadoop.io.BytesWritable;
import org.apache.log4j.Logger;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.MatOfByte;
import org.opencv.core.Rect;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;
import org.opencv.utils.Converters;

public class Utils {

	public static Logger l = Logger.getLogger(Test.class);

	public static byte[] matToByteArr(Mat m) {
		MatOfByte buf = new MatOfByte();
		Imgcodecs.imencode(".jpg", m, buf);
		m.convertTo(m, CvType.CV_8UC1);
		byte[] b = buf.toArray();
		return b;
	}

	/* HELPER FUNCTIONS */
	public static Mat byteswritableToOpenCVMat(BytesWritable inputBW) {
		byte[] imageFileBytes = inputBW.getBytes();
		Mat img = new Mat();
		Byte[] bigByteArray = new Byte[imageFileBytes.length];
		for (int i = 0; i < imageFileBytes.length; i++)
			bigByteArray[i] = new Byte(imageFileBytes[i]);
		List<Byte> matlist = Arrays.asList(bigByteArray);
		img = Converters.vector_char_to_Mat(matlist);
		img = Imgcodecs.imdecode(img, Imgcodecs.CV_LOAD_IMAGE_COLOR);
		return img;
	}

	public static String byteswritableToString(BytesWritable inputBW) {
		byte[] imageFileBytes = inputBW.getBytes();
		String metadataString = new String(imageFileBytes, 0,
				imageFileBytes.length,
				java.nio.charset.Charset.forName("ISO-8859-1"));
		return metadataString;
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
