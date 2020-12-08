package test_dl;

import org.datavec.api.records.reader.RecordReader;
import org.datavec.api.records.reader.impl.csv.CSVRecordReader;
import org.datavec.api.split.FileSplit;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.nd4j.common.io.ClassPathResource;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;

import java.io.IOException;

public class Main
{
    private static final int FEATURES_COUNT = 2;
    private static final int CLASSES_COUNT = 3;

    public static void main(String[] args ) throws IOException, InterruptedException {
        //https://www.baeldung.com/deeplearning4j
        try (RecordReader recordReader = new CSVRecordReader(0, ',')) {
            recordReader.initialize(new FileSplit(
                    new ClassPathResource("../data/iris.data").getFile()));

            DataSetIterator iterator = new RecordReaderDataSetIterator(
                    recordReader, 150, FEATURES_COUNT, CLASSES_COUNT);
            DataSet allData = iterator.next();
            allData.shuffle(42);
        }    }
}
