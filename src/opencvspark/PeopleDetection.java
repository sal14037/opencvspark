package opencvspark;
import java.util.List;

import org.apache.log4j.Level;
import org.apache.log4j.Logger;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.MatOfDouble;
import org.opencv.core.MatOfFloat;
import org.opencv.core.MatOfRect;
import org.opencv.core.Rect;
import org.opencv.core.Scalar;
import org.opencv.objdetect.CascadeClassifier;
import org.opencv.objdetect.HOGDescriptor;

public class PeopleDetection {

	public Logger l = Logger.getLogger(this.getClass());

	public PeopleDetection() {
		l.setLevel(Level.INFO);
	}

	// initialise HoG Descriptor with default values
	final HOGDescriptor hog = new HOGDescriptor();
	final MatOfFloat descriptors = HOGDescriptor.getDefaultPeopleDetector();

	// necessary matrices for descriptor
	private final MatOfRect foundLocations = new MatOfRect();
	private final MatOfDouble foundWeights = new MatOfDouble();
	private List<Double> weightList;
	private List<Rect> rectList;
	private int numberOfPeople = 0;
	final Scalar rectColor = new Scalar(0, 255, 0);
	final Scalar fontColor = new Scalar(255, 255, 255);
	private double threshold = 0.96;
	private boolean allwrong = true;

	public Mat detectPeople(Mat m) {
		l.info("People detection");
		// Prepare People detection
		m.convertTo(m, CvType.CV_8UC3);
		hog.setSVMDetector(descriptors);
		hog.detectMultiScale(m, foundLocations, foundWeights);
		weightList = getFoundWeights().toList();
		rectList = getFoundLocations().toList();
		return m;
	}

	public List<Double> getWeightList() {
		return weightList;
	}

	public void setWeightList(List<Double> weightList) {
		this.weightList = weightList;
	}

	public List<Rect> getRectList() {
		return rectList;
	}

	public void setRectList(List<Rect> rectList) {
		this.rectList = rectList;
	}

	public Scalar getRectColor() {
		return rectColor;
	}

	public Scalar getFontColor() {
		return fontColor;
	}

	public int getNumberOfDetectedPeople() {
		if (allwrong)
			return 0;
		numberOfPeople = foundLocations.rows();
		return numberOfPeople;
	}

	public boolean allIsLost() {
		for (Double d : foundWeights.toList()) {
			if (d >= threshold) {
				allwrong = false;
				break;
			}
		}
		return allwrong;
	}

	public boolean withinThreshold(int index) {
		if (foundWeights.toList().get(index).doubleValue() < threshold)
			return false;
		else
			return true;
	}

	public int getFramesWithPeople() {
		return numberOfPeople;
	}

	public double getThreshold() {
		return threshold;
	}

	public void setThreshold(double threshold) {
		this.threshold = threshold;
	}

	public MatOfRect getFoundLocations() {
		return foundLocations;
	}

	public MatOfDouble getFoundWeights() {
		return foundWeights;
	}

}
