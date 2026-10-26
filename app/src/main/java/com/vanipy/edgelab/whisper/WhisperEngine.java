package com.vanipy.edgelab.whisper;

import java.io.File;
import java.nio.FloatBuffer;
import java.nio.LongBuffer;
import java.nio.file.Files;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;

import ai.onnxruntime.OnnxTensor;
import ai.onnxruntime.OrtEnvironment;
import ai.onnxruntime.OrtSession;

import org.json.JSONObject;

/**
 * Whisper-tiny (quantized ONNX, onnx-community export) greedy transcription
 * with KV cache. All on-device.
 */
public class WhisperEngine implements AutoCloseable {

    public static class Result {
        public String text;
        public int tokens;
        public long melMs, encodeMs, decodeMs;
    }

    private final OrtEnvironment env;
    private final OrtSession encoder;
    private final OrtSession decoderInit;
    private final OrtSession decoderPast;
    private final WhisperTokenizer tokenizer;

    private final int sot;
    private final int eot;
    private final int transcribe;
    private final int noTimestamps;
    private final Set<Integer> suppress = new HashSet<>();
    private final Set<Integer> suppressBegin = new HashSet<>();

    private static final int LAYERS = 4;
    private static final int HEADS = 6;
    private static final int HEAD_DIM = 64;
    private static final int VOCAB = 51865;
    private static final int MAX_TOKENS = 224;
    private static final int FIRST_TIMESTAMP_TOKEN = 50364;

    public WhisperEngine(File dir, java.io.InputStream melFilters) throws Exception {
        MelSpectrogram.loadFilters(melFilters);
        env = OrtEnvironment.getEnvironment();
        OrtSession.SessionOptions opts = new OrtSession.SessionOptions();
        encoder = env.createSession(new File(dir, "whisper_encoder.onnx").getAbsolutePath(), opts);
        decoderInit = env.createSession(new File(dir, "whisper_decoder_init.onnx").getAbsolutePath(), opts);
        decoderPast = env.createSession(new File(dir, "whisper_decoder_past.onnx").getAbsolutePath(), opts);
        tokenizer = new WhisperTokenizer(new File(dir, "whisper_tokenizer.json").getPath());

        sot = tokenizer.specialId("<|startoftranscript|>", 50258);
        eot = tokenizer.specialId("<|endoftext|>", 50257);
        transcribe = tokenizer.specialId("<|transcribe|>", 50359);
        noTimestamps = tokenizer.specialId("<|notimestamps|>", 50363);

        try {
            String gc = new String(Files.readAllBytes(
                    new File(dir, "whisper_generation_config.json").toPath()));
            JSONObject g = new JSONObject(gc);
            org.json.JSONArray st = g.getJSONArray("suppress_tokens");
            for (int i = 0; i < st.length(); i++) suppress.add(st.getInt(i));
            org.json.JSONArray bs = g.getJSONArray("begin_suppress_tokens");
            for (int i = 0; i < bs.length(); i++) suppressBegin.add(bs.getInt(i));
        } catch (Exception ignored) {}
    }

    public int languageToken(String content, int fallback) {
        return tokenizer.specialId(content, fallback);
    }

    public Result transcribe(float[] pcm) throws Exception {
        Result result = new Result();

        long t0 = System.nanoTime();
        float[][] mel = MelSpectrogram.logMel(pcm);
        result.melMs = (System.nanoTime() - t0) / 1_000_000;

        float[] flat = new float[80 * 3000];
        for (int m = 0; m < 80; m++) System.arraycopy(mel[m], 0, flat, m * 3000, 3000);

        t0 = System.nanoTime();
        float[] hiddenFlat;
        try (OnnxTensor in = OnnxTensor.createTensor(env, FloatBuffer.wrap(flat), new long[]{1, 80, 3000});
             OrtSession.Result er = encoder.run(java.util.Collections.singletonMap("input_features", in))) {
            float[][][] hidden = (float[][][]) er.get(0).getValue();
            hiddenFlat = new float[1500 * 384];
            for (int i = 0; i < 1500; i++) {
                System.arraycopy(hidden[0][i], 0, hiddenFlat, i * 384, 384);
            }
        }
        result.encodeMs = (System.nanoTime() - t0) / 1_000_000;

        t0 = System.nanoTime();
        List<Integer> prefix = new ArrayList<>();
        prefix.add(sot);
        // language token chosen by caller via transcribe(pcm, langToken) below
        result.decodeMs = decodeInternal(prefix, hiddenFlat, result);
        return result;
    }

    public Result transcribe(float[] pcm, int langToken) throws Exception {
        Result result = new Result();
        long t0 = System.nanoTime();
        float[][] mel = MelSpectrogram.logMel(pcm);
        result.melMs = (System.nanoTime() - t0) / 1_000_000;

        float[] flat = new float[80 * 3000];
        for (int m = 0; m < 80; m++) System.arraycopy(mel[m], 0, flat, m * 3000, 3000);

        t0 = System.nanoTime();
        float[] hiddenFlat;
        try (OnnxTensor in = OnnxTensor.createTensor(env, FloatBuffer.wrap(flat), new long[]{1, 80, 3000});
             OrtSession.Result er = encoder.run(java.util.Collections.singletonMap("input_features", in))) {
            float[][][] hidden = (float[][][]) er.get(0).getValue();
            hiddenFlat = new float[1500 * 384];
            for (int i = 0; i < 1500; i++) {
                System.arraycopy(hidden[0][i], 0, hiddenFlat, i * 384, 384);
            }
        }
        result.encodeMs = (System.nanoTime() - t0) / 1_000_000;

        List<Integer> prefix = new ArrayList<>();
        prefix.add(sot);
        prefix.add(langToken);
        prefix.add(transcribe);
        prefix.add(noTimestamps);

        t0 = System.nanoTime();
        decodeInternal(prefix, hiddenFlat, result);
        result.decodeMs = (System.nanoTime() - t0) / 1_000_000;
        return result;
    }

    private long decodeInternal(List<Integer> prefix, float[] hiddenFlat, Result result) throws Exception {
        List<Integer> generated = new ArrayList<>();
        OrtSession.Result firstResult = null;
        OrtSession.Result prevResult = null;
        try {
            long[] prefixArr = new long[prefix.size()];
            for (int i = 0; i < prefix.size(); i++) prefixArr[i] = prefix.get(i);

            // First pass: decoder without past, over the full prefix.
            // Results stay OPEN: present.* tensors reused as KV cache are owned by them.
            OnnxTensor ids = OnnxTensor.createTensor(env, LongBuffer.wrap(prefixArr), new long[]{1, prefixArr.length});
            OnnxTensor hidden = OnnxTensor.createTensor(env, FloatBuffer.wrap(hiddenFlat), new long[]{1, 1500, 384});
            Map<String, OnnxTensor> inputs = new HashMap<>();
            inputs.put("input_ids", ids);
            inputs.put("encoder_hidden_states", hidden);
            firstResult = decoderInit.run(inputs);
            prevResult = firstResult;
            ids.close();
            hidden.close();

            int next = argmaxLast(firstResult, true);
            Map<String, OnnxTensor> encPast = new HashMap<>();
            Map<String, OnnxTensor> decPast = new HashMap<>();
            splitPresent(firstResult, encPast, decPast);

            // Subsequent passes: decoder with past, one token at a time.
            // Encoder cross-KV is fixed after the first pass; only decoder self-KV advances.
            while (next != eot && generated.size() < MAX_TOKENS) {
                generated.add(next);
                Map<String, OnnxTensor> step = new HashMap<>();
                step.putAll(decPast);
                step.putAll(encPast);
                long[] one = {next};
                OnnxTensor ids1 = OnnxTensor.createTensor(env, LongBuffer.wrap(one), new long[]{1, 1});
                step.put("input_ids", ids1);
                OrtSession.Result stepResult = decoderPast.run(step);
                ids1.close();

                next = argmaxLast(stepResult, false);
                Map<String, OnnxTensor> newDecPast = new HashMap<>();
                for (Map.Entry<String, ? extends ai.onnxruntime.OnnxValue> e : stepResult) {
                    if (e.getKey().startsWith("present.") && e.getKey().contains(".decoder.")) {
                        newDecPast.put(e.getKey().replace("present.", "past_key_values."), (OnnxTensor) e.getValue());
                    }
                }
                // The new run has consumed the previous result's tensors; it can be closed now.
                // firstResult owns encPast for the whole loop, so it is closed only at the end.
                if (prevResult != firstResult) {
                    try { prevResult.close(); } catch (Exception ignored) {}
                }
                prevResult = stepResult;
                decPast = newDecPast;
            }
        } finally {
            if (prevResult != null && prevResult != firstResult) {
                try { prevResult.close(); } catch (Exception ignored) {}
            }
            if (firstResult != null) {
                try { firstResult.close(); } catch (Exception ignored) {} finally { prevResult = null; }
            }
        }

        result.text = tokenizer.decode(generated).trim();
        result.tokens = generated.size();
        return 0;
    }

    private int argmaxLast(OrtSession.Result dr, boolean first) throws Exception {
        OnnxTensor logitsTensor = (OnnxTensor) dr.get("logits").get();
        float[][][] logits = (float[][][]) logitsTensor.getValue();
        float[] last = logits[0][logits[0].length - 1];
        int best = -1;
        float bestV = Float.NEGATIVE_INFINITY;
        for (int i = 0; i < Math.min(last.length, VOCAB); i++) {
            if (suppress.contains(i)) continue;
            if (first && suppressBegin.contains(i)) continue;
            if (i >= FIRST_TIMESTAMP_TOKEN) continue; // no-timestamps mode
            if (last[i] > bestV) { bestV = last[i]; best = i; }
        }
        return best < 0 ? eot : best;
    }

    private void splitPresent(OrtSession.Result dr, Map<String, OnnxTensor> encPast, Map<String, OnnxTensor> decPast) {
        for (Map.Entry<String, ? extends ai.onnxruntime.OnnxValue> e : dr) {
            if (!e.getKey().startsWith("present.")) continue;
            String name = e.getKey().replace("present.", "past_key_values.");
            if (e.getKey().contains(".encoder.")) encPast.put(name, (OnnxTensor) e.getValue());
            else decPast.put(name, (OnnxTensor) e.getValue());
        }
    }

    @Override
    public void close() {
        try { encoder.close(); } catch (Exception ignored) {}
        try { decoderInit.close(); } catch (Exception ignored) {}
        try { decoderPast.close(); } catch (Exception ignored) {}
    }
}
