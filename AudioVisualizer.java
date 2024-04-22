import javafx.application.Application;
import javafx.scene.Scene;
import javafx.scene.chart.NumberAxis;
import javafx.scene.chart.XYChart;
import javafx.scene.chart.LineChart;
import javafx.stage.Stage;

import javax.sound.sampled.*;
import java.io.File;
import java.io.IOException;

public class AudioVisualizer extends Application {

    public double[] readAudioFile(String filePath) {
        try {
            File file = new File(filePath);
            AudioInputStream stream = AudioSystem.getAudioInputStream(file);
            AudioFormat format = stream.getFormat();
            int frameSize = format.getFrameSize();
            int frames = (int) (stream.getFrameLength());
            byte[] bytes = new byte[frames * frameSize];

            int bytesRead = stream.read(bytes);
            double[] data = new double[bytes.length / 2]; // Assuming 16-bit samples

            for (int i = 0; i < data.length; i++) {
                // Convert two bytes to a single integer
                int byteIndex = i * 2;
                int amplitude = ((bytes[byteIndex + 1] & 0xFF) << 8) | (bytes[byteIndex] & 0xFF);
                data[i] = amplitude;
            }

            stream.close();
            return data;
        } catch (UnsupportedAudioFileException | IOException ex) {
            ex.printStackTrace();
            return null; // or handle more gracefully
        }
    }

    @Override
    public void start(Stage primaryStage) {
        primaryStage.setTitle("Audio Waveform Visualizer");

        final NumberAxis xAxis = new NumberAxis();
        final NumberAxis yAxis = new NumberAxis();
        xAxis.setLabel("Sample");
        yAxis.setLabel("Amplitude");

        final LineChart<Number, Number> lineChart = new LineChart<>(xAxis, yAxis);
        lineChart.setTitle("Waveform of '09 Purple Rain.wav'");

        XYChart.Series<Number, Number> series = new XYChart.Series<>();
        series.setName("Audio Data");

        double[] audioData = readAudioFile("09 Purple Rain.wav"); // Adjust the file path as necessary
        if (audioData != null) {
            for (int i = 0; i < audioData.length; i++) {
                series.getData().add(new XYChart.Data<>(i, audioData[i]));
            }
        }

        lineChart.getData().add(series);
        Scene scene = new Scene(lineChart, 800, 600);
        primaryStage.setScene(scene);
        primaryStage.show();
    }

    public static void main(String[] args) {
        launch(args);
    }
}
