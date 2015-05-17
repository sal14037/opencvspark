package opencvspark;

import org.apache.spark.api.java.*;
import org.apache.spark.SparkConf;
import org.apache.spark.api.java.function.Function;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.*;
import org.apache.hadoop.mapred.FileInputFormat;
import org.apache.hadoop.mapreduce.Mapper.Context;
import org.apache.log4j.Level;
import org.apache.log4j.Logger;

import scala.Tuple2;
import hipi.image.FloatImage;
import hipi.image.ImageHeader;
import hipi.imagebundle.mapreduce.ImageBundleInputFormat;

import java.io.IOException;
import java.util.Date;
import java.util.List;
import java.util.Random;

import org.apache.spark.api.java.function.PairFunction;
import org.opencv.core.Core;
import org.opencv.core.Mat;
import org.opencv.core.Point;
import org.opencv.core.Rect;
import org.opencv.core.Scalar;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;
import org.opencv.utils.Converters;

import java.util.Arrays;

import org.apache.commons.io.FilenameUtils;
import org.json.simple.*;
import org.json.simple.parser.JSONParser;

public class Test {

	static {
		System.loadLibrary(Core.NATIVE_LIBRARY_NAME);
	}

	public static void main(String[] args) {
		String imagesSequenceFile = "/home/thomas/test.hsf";
		SparkConf conf = new SparkConf().setAppName("opencvspark").setMaster(
				"local[4]");
		JavaSparkContext sc = new JavaSparkContext(conf);
		// 1) get a non serializable RDD with the images
		JavaPairRDD<Text, BytesWritable> imagesRDD_ = sc.sequenceFile(
				imagesSequenceFile, Text.class, BytesWritable.class);

		// 2) get a serializable RDD with the information extracted from the
		// images
		JavaPairRDD<String, String> imagesRDD = imagesRDD_
				.mapToPair(new PairFunction<Tuple2<Text, BytesWritable>, String, String>() {
					public Tuple2<String, String> call(
							Tuple2<Text, BytesWritable> kv) throws Exception {
						String filename = kv._1.toString();
						String filenameWithoutExtension = FilenameUtils
								.getName(filename);
						Mat cv = byteswritableToOpenCVMat(kv._2);

						// Init People detection
						PeopleDetection p = new PeopleDetection();
						int framesWithPeople = 0;
						Point rectPoint1 = new Point();
						Point rectPoint2 = new Point();
						Point fontPoint = new Point();

						// Init Face detection
						FaceDetection fd = new FaceDetection();
						Mat mRgba = new Mat();
						Mat mGrey = new Mat();

						// Init Face recognition
						FaceRecognition fr = new FaceRecognition();
						Mat face = new Mat();

						if (!cv.empty()) {
							p.detectPeople(cv);
							if (p.getFoundLocations().rows() > 0) {
								int i = 0;

								// copy input_image to other matrices
								Mat toCrop = new Mat();
								cv.copyTo(toCrop);

								if (!p.allIsLost()) {
									for (Rect rect : p.getRectList()) {
										// saving the positions of the detected
										// person in a
										// Point
										rectPoint1.x = rect.x;
										rectPoint1.y = rect.y;
										rectPoint2.x = rect.x + rect.width;
										rectPoint2.y = rect.y + rect.height;
										// location for text
										fontPoint.x = rect.x;
										fontPoint.y = rect.y - 4;
										// CHECKSTYLE:ON MagicNumber
										if (p.withinThreshold(i)) {
											framesWithPeople++;
											// crop image area
											mRgba = Utils.cropMatrix(toCrop,
													rect);
											mGrey = fd.enhance(mRgba);

											// detect faces
											fd.detectFace(mGrey);
											if (!fd.faces.empty()) {
												for (Rect rect1 : fd.faces
														.toArray()) {
													face = Utils.cropMatrix(
															mGrey, rect1);
													fr.loadRecogniser();
													fr.recognise(face);
													if (fr.getLabel() > 0) {
														// draw rectangle plus
														// text
														Imgproc.putText(
																cv,
																"Person "
																		+ fr.getLabel(),
																new Point(
																		rect.x
																				+ rect1.x,
																		rect.y
																				+ rect1.y
																				- 5),
																Core.FONT_HERSHEY_PLAIN,
																1,
																fr.getColour(),
																2,
																Core.LINE_AA,
																false);
														Imgproc.rectangle(
																cv,
																new Point(
																		rect.x
																				+ rect1.x,
																		rect.y
																				+ rect1.y),
																new Point(
																		rect.x
																				+ rect1.x
																				+ rect1.width,
																		rect.y
																				+ rect1.y
																				+ rect1.height),
																fr.getColour(),
																2);
													} else {
														Imgproc.putText(
																cv,
																"?",
																new Point(
																		rect.x
																				+ rect1.x,
																		rect.y
																				+ rect1.y
																				- 5),
																Core.FONT_HERSHEY_PLAIN,
																1, new Scalar(
																		0, 255,
																		255),
																2,
																Core.LINE_AA,
																false);
														Imgproc.rectangle(
																cv,
																new Point(
																		rect.x
																				+ rect1.x,
																		rect.y
																				+ rect1.y),
																new Point(
																		rect.x
																				+ rect1.x
																				+ rect1.width,
																		rect.y
																				+ rect1.y
																				+ rect1.height),
																new Scalar(0,
																		255,
																		255), 2);
													}
												}
											}
											// Draw rectangle around found
											// object
											Imgproc.rectangle(cv, rectPoint1,
													rectPoint2, p.rectColor, 2);
											Imgproc.putText(cv, String.format(
													"%1.2f", p.getWeightList()
															.get(i)),
													fontPoint,
													Core.FONT_HERSHEY_PLAIN,
													1.5, p.rectColor, 2,
													Core.LINE_AA, false);
										}
										if (i < p.getWeightList().size() - 1) {
											i++;
										}
									}
								} else {
									// if no person is detected
									mGrey = fd.enhance(cv);
									fd.detectFace(mGrey);
									if (fd.faces.rows() > 0) {
										for (Rect rect1 : fd.faces.toArray()) {
											face = Utils.cropMatrix(mGrey,
													rect1);
											fr.loadRecogniser();
											fr.recognise(face);
											Imgproc.rectangle(
													cv,
													new Point(rect1.x, rect1.y),
													new Point(
															rect1.x
																	+ rect1.width,
															rect1.y
																	+ rect1.height),
													new Scalar(0, 255, 255), 2);
											Imgproc.putText(cv, "?", new Point(
													rect1.x, rect1.y - 5),
													Core.FONT_HERSHEY_PLAIN, 1,
													new Scalar(0, 255, 255), 2,
													Core.LINE_AA, false);
										}
									}
								}
							} else {
								// if no person is detected
								mGrey = fd.enhance(cv);
								fd.detectFace(mGrey);
								if (fd.faces.rows() > 0) {
									for (Rect rect1 : fd.faces.toArray()) {
										face = Utils.cropMatrix(mGrey, rect1);
										fr.loadRecogniser();
										fr.recognise(face);
										Imgproc.rectangle(cv, new Point(
												rect1.x, rect1.y), new Point(
												rect1.x + rect1.width, rect1.y
														+ rect1.height),
												new Scalar(0, 255, 255), 2);
										Imgproc.putText(cv, "?", new Point(
												rect1.x, rect1.y - 5),
												Core.FONT_HERSHEY_PLAIN, 1,
												new Scalar(0, 255, 255), 2,
												Core.LINE_AA, false);
									}
								}
							}
							Imgproc.putText(cv, "TEST", new Point(
									cv.width() - 380, cv.height() - 10),
									Core.FONT_HERSHEY_PLAIN, 1.5, p.fontColor,
									2, Core.LINE_AA, false);
						}
						Random r = new Random();
						return new Tuple2(filenameWithoutExtension, "Camera"
								+ r.nextInt(10) + cv.cols() + "Label:"
								+ fr.getLabel());
					}
				});

		List<Tuple2<String, String>> output3 = imagesRDD.collect();
		for (Tuple2<String, String> tuple : output3) {
			System.out.println(tuple._1() + ": " + tuple._2());
		}
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
}
