package com.vanipy.edgelab.ort;

import android.content.Context;
import android.os.Handler;
import android.os.Looper;

import java.io.File;
import java.io.FileOutputStream;
import java.io.InputStream;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

import okhttp3.OkHttpClient;
import okhttp3.Request;
import okhttp3.Response;

/**
 * Downloads model files on explicit user request only, into app-private storage.
 * No background fetching, no uploads.
 */
public class ModelStore {

    public interface Callback {
        void onProgress(long done, long total);
        void onDone(File dir);
        void onError(String message);
    }

    public static class RemoteFile {
        public final String name;
        public final String url;
        public final long expectedBytes;
        public RemoteFile(String name, String url, long expectedBytes) {
            this.name = name; this.url = url; this.expectedBytes = expectedBytes;
        }
    }

    public static final RemoteFile[] WHISPER = {
            new RemoteFile("whisper_encoder.onnx",
                    "https://huggingface.co/onnx-community/whisper-tiny/resolve/main/onnx/encoder_model_quantized.onnx",
                    10124990L),
            new RemoteFile("whisper_decoder_init.onnx",
                    "https://huggingface.co/onnx-community/whisper-tiny/resolve/main/onnx/decoder_model_quantized.onnx",
                    30461071L),
            new RemoteFile("whisper_decoder_past.onnx",
                    "https://huggingface.co/onnx-community/whisper-tiny/resolve/main/onnx/decoder_with_past_model_quantized.onnx",
                    29220476L),
            new RemoteFile("whisper_tokenizer.json",
                    "https://huggingface.co/onnx-community/whisper-tiny/resolve/main/tokenizer.json",
                    2480466L),
            new RemoteFile("whisper_generation_config.json",
                    "https://huggingface.co/onnx-community/whisper-tiny/resolve/main/generation_config.json",
                    3772L),
    };

    public static final RemoteFile[] MOBILENET = {
            new RemoteFile("mobilenetv2.onnx",
                    "https://media.githubusercontent.com/media/onnx/models/main/validated/vision/classification/mobilenet/model/mobilenetv2-7.onnx",
                    14246826L),
    };

    public static long totalBytes(RemoteFile[] files) {
        long t = 0;
        for (RemoteFile f : files) t += f.expectedBytes;
        return t;
    }

    public static File dir(Context ctx) {
        File d = new File(ctx.getFilesDir(), "models");
        d.mkdirs();
        return d;
    }

    public static boolean isDownloaded(Context ctx, RemoteFile[] files) {
        File d = dir(ctx);
        for (RemoteFile f : files) {
            File local = new File(d, f.name);
            if (!local.exists() || (f.expectedBytes > 0 && local.length() != f.expectedBytes)) return false;
        }
        return true;
    }

    public static File file(Context ctx, String name) {
        return new File(dir(ctx), name);
    }

    public static void delete(Context ctx, RemoteFile[] files) {
        for (RemoteFile f : files) new File(dir(ctx), f.name).delete();
    }

    public static void download(Context ctx, RemoteFile[] files, Callback cb) {
        Handler main = new Handler(Looper.getMainLooper());
        ExecutorService ex = Executors.newSingleThreadExecutor();
        ex.execute(() -> {
            OkHttpClient client = new OkHttpClient();
            try {
                for (RemoteFile f : files) {
                    File out = new File(dir(ctx), f.name);
                    if (out.exists() && (f.expectedBytes <= 0 || out.length() == f.expectedBytes)) continue;
                    File tmp = new File(dir(ctx), f.name + ".part");
                    Request req = new Request.Builder().url(f.url).build();
                    try (Response resp = client.newCall(req).execute()) {
                        if (!resp.isSuccessful() || resp.body() == null) {
                            throw new Exception("HTTP " + resp.code() + " for " + f.name);
                        }
                        long total = f.expectedBytes > 0 ? f.expectedBytes : resp.body().contentLength();
                        try (InputStream in = resp.body().byteStream();
                             FileOutputStream fos = new FileOutputStream(tmp)) {
                            byte[] buf = new byte[64 * 1024];
                            long done = 0;
                            int n;
                            long lastPost = 0;
                            while ((n = in.read(buf)) != -1) {
                                fos.write(buf, 0, n);
                                done += n;
                                if (done - lastPost > 512 * 1024) {
                                    lastPost = done;
                                    long fd = done, ft = total;
                                    main.post(() -> cb.onProgress(fd, ft));
                                }
                            }
                        }
                    }
                    if (f.expectedBytes > 0 && tmp.length() != f.expectedBytes) {
                        tmp.delete();
                        throw new Exception("Size mismatch for " + f.name + ": got " + tmp.length()
                                + ", expected " + f.expectedBytes);
                    }
                    tmp.renameTo(out);
                }
                File d = dir(ctx);
                main.post(() -> cb.onDone(d));
            } catch (Exception e) {
                main.post(() -> cb.onError(e.getMessage() == null ? e.toString() : e.getMessage()));
            } finally {
                ex.shutdown();
            }
        });
    }
}
