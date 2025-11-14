package com.example.authentitk;

import androidx.annotation.Nullable;
import androidx.appcompat.app.AppCompatActivity;

import android.Manifest;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.graphics.Bitmap;
import android.media.ThumbnailUtils;
import android.net.Uri;
import android.os.Bundle;
import android.provider.MediaStore;
import android.view.View;
import androidx.appcompat.widget.AppCompatButton;

import android.view.WindowManager;
import android.widget.ImageView;
import android.widget.TextView;


import org.tensorflow.lite.DataType;
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;

import com.example.authentitk.ml.Model2;
import com.google.android.gms.auth.api.signin.GoogleSignIn;
import com.google.android.gms.auth.api.signin.GoogleSignInAccount;
import com.google.android.gms.auth.api.signin.GoogleSignInClient;
import com.google.android.gms.auth.api.signin.GoogleSignInOptions;


import com.example.authentitk.ml.Model;


public class Authenti_TK extends AppCompatActivity {

    GoogleSignInOptions gso;
    GoogleSignInClient gsc;
    TextView name,result;
    AppCompatButton sign_out,capture,gallery;
    ImageView imageView;

    int imageSize = 224;
    int width = 256;
    int height = 117;

    private Model2 model2;
    private Model model;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_home_page);
        getWindow().setFlags(WindowManager.LayoutParams.FLAG_FULLSCREEN,WindowManager.LayoutParams.FLAG_FULLSCREEN);

        gso=new GoogleSignInOptions.Builder(GoogleSignInOptions.DEFAULT_SIGN_IN).requestEmail().build();
        gsc= GoogleSignIn.getClient(this,gso);
        name=findViewById(R.id.name);
        sign_out=findViewById(R.id.sign_out);
        capture=findViewById(R.id.capture_photo);
        gallery=findViewById(R.id.gallery);

        result= findViewById(R.id.result);
        imageView= findViewById(R.id.imageView);

        GoogleSignInAccount account = GoogleSignIn.getLastSignedInAccount(this);

        if (account != null) {
            // Access the user's name and display it in the TextView
            String userName = account.getDisplayName();
            name.setText(userName);
        }
        sign_out.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                signOut(); // Call the signOut method
            }
        });
        capture.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                if (checkSelfPermission(Manifest.permission.CAMERA) == PackageManager.PERMISSION_GRANTED) {
                    Intent cameraIntent = new Intent(MediaStore.ACTION_IMAGE_CAPTURE);
                    startActivityForResult(cameraIntent, 3);
                } else {
                    requestPermissions(new String[]{Manifest.permission.CAMERA}, 100);
                }
            }
        });
        gallery.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                Intent cameraIntent = new Intent(Intent.ACTION_PICK, MediaStore.Images.Media.EXTERNAL_CONTENT_URI);
                startActivityForResult(cameraIntent, 1);
            }
        });

        try {
            model2 = Model2.newInstance(getApplicationContext());;
            model = Model.newInstance(getApplicationContext());
        } catch (IOException e) {
            // Handle model loading errors here.
            e.printStackTrace();
        }

    }
    private void signOut() {
        // Sign the user out and navigate to the sign-in activity
        gsc.signOut().addOnCompleteListener(this, task -> {
            // After signing out, navigate to the main activity or another suitable screen
            startActivity(new Intent(Authenti_TK.this, MainActivity.class));
            finish();
        });
    }

    public void process(Bitmap image){
        try{
            // Creates inputs for reference.
            TensorBuffer inputFeature0 = TensorBuffer.createFixedSize(new int[]{1, 256, 117, 3}, DataType.FLOAT32);
            ByteBuffer byteBuffer = ByteBuffer.allocateDirect(4 * width * height * 3);
            byteBuffer.order(ByteOrder.nativeOrder());

            int[] intValues = new int[width * height];
            image.getPixels(intValues, 0, image.getWidth(), 0, 0, image.getWidth(), image.getHeight());
            int pixel = 0;

            // Iterate over each pixel and extract R, G, and B values. Add those values individually to the byte buffer.
            for (int i = 0; i < width; i++) {
                for (int j = 0; j < height; j++) {
                    int val = intValues[pixel++]; // RGB
                    byteBuffer.putFloat(((val >> 16) & 0xFF) * (1.f / 1));
                    byteBuffer.putFloat(((val >> 8) & 0xFF) * (1.f / 1));
                    byteBuffer.putFloat((val & 0xFF) * (1.f / 1));
                }
            }

            inputFeature0.loadBuffer(byteBuffer);

            // Runs model inference and gets result.
            Model2.Outputs outputs = model2.process(inputFeature0);
            TensorBuffer outputFeature0 = outputs.getOutputFeature0AsTensorBuffer();

            float[] confidences = outputFeature0.getFloatArray();
            // find the index of the class with the biggest confidence.
            int maxPos = 0;
            float maxConfidence1 = 0;
            for (int i = 0; i < confidences.length; i++) {
                if (confidences[i] > maxConfidence1) {
                    maxConfidence1 = confidences[i];
                    maxPos = i;
                }
            }
            String[] classes1 = {"10", "100", "1000", "20", "5", "50", "500"};

            if (classes1[maxPos].equals("500") || classes1[maxPos].equals("1000")) {
                image = Bitmap.createScaledBitmap(image, imageSize, imageSize, false);

                // Creates inputs for reference.
                TensorBuffer inputFeature1 = TensorBuffer.createFixedSize(new int[]{1, 224, 224, 3}, DataType.FLOAT32);
                ByteBuffer byteBuffer1 = ByteBuffer.allocateDirect(4 * imageSize * imageSize * 3);
                byteBuffer1.order(ByteOrder.nativeOrder());

                int[] intValues1 = new int[imageSize * imageSize];
                image.getPixels(intValues1, 0, image.getWidth(), 0, 0, image.getWidth(), image.getHeight());
                int pixel1 = 0;

                // Iterate over each pixel and extract R, G, and B values. Add those values individually to the byte buffer.
                for (int i = 0; i < imageSize; i++) {
                    for (int j = 0; j < imageSize; j++) {
                        int val = intValues1[pixel1++]; // RGB
                        byteBuffer1.putFloat(((val >> 16) & 0xFF) * (1.f / 1));
                        byteBuffer1.putFloat(((val >> 8) & 0xFF) * (1.f / 1));
                        byteBuffer1.putFloat((val & 0xFF) * (1.f / 1));
                    }
                }

                inputFeature1.loadBuffer(byteBuffer1);

                // Runs model inference and gets result.
                Model.Outputs outputs1 = model.process(inputFeature1);
                TensorBuffer outputFeature1 = outputs1.getOutputFeature0AsTensorBuffer();

                float[] confidencesModel = outputFeature1.getFloatArray();
                // find the index of the class with the biggest confidence.
                int maxPosModel = 0;
                float maxConfidenceModel = 0;
                for (int i = 0; i < confidencesModel.length; i++) {
                    if (confidencesModel[i] > maxConfidenceModel) {
                        maxConfidenceModel = confidencesModel[i];
                        maxPosModel = i;
                    }
                }
                String[] classes2 = {"Fake 1000TK", "Fake 500TK", "Real 1000TK", "Real 500TK"};
                result.setText(classes2[maxPosModel]);
                model.close();
            } else {
                result.setText(classes1[maxPos]);
            }

            // Releases model resources if no longer used.
            model2.close();
        }catch (Exception e) {
            e.printStackTrace();
        }
    }

    @Override
    protected void onActivityResult(int requestCode, int resultCode, @Nullable Intent data) {
        if(resultCode == RESULT_OK){
            if(requestCode == 3){
                Bitmap image = (Bitmap) data.getExtras().get("data");
                int dimension = Math.min(image.getWidth(), image.getHeight());
                image = ThumbnailUtils.extractThumbnail(image, dimension, dimension);
                imageView.setImageBitmap(image);
                image = Bitmap.createScaledBitmap(image, width, height, false);
                process(image);
            }else{
                Uri dat = data.getData();
                Bitmap image = null;
                try {
                    image = MediaStore.Images.Media.getBitmap(this.getContentResolver(), dat);
                } catch (IOException e) {
                    e.printStackTrace();
                }
                imageView.setImageBitmap(image);

                image = Bitmap.createScaledBitmap(image, width, height, false);
                process(image);
            }
        }
        super.onActivityResult(requestCode, resultCode, data);
    }
}