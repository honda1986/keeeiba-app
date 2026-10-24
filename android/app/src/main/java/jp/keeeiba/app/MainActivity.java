package jp.keeeiba.app;

import android.app.Activity;
import android.content.ClipData;
import android.content.ClipboardManager;
import android.content.Context;
import android.content.Intent;
import android.content.SharedPreferences;
import android.graphics.Color;
import android.graphics.Typeface;
import android.net.Uri;
import android.os.Bundle;
import android.os.Handler;
import android.os.Looper;
import android.os.SystemClock;
import android.view.Gravity;
import android.view.InputDevice;
import android.view.MotionEvent;
import android.view.View;
import android.view.ViewGroup;
import android.webkit.CookieManager;
import android.webkit.JavascriptInterface;
import android.webkit.WebChromeClient;
import android.webkit.WebResourceRequest;
import android.webkit.WebSettings;
import android.webkit.WebView;
import android.webkit.WebViewClient;
import android.widget.Button;
import android.widget.FrameLayout;
import android.widget.LinearLayout;
import android.widget.TextView;
import android.widget.Toast;

import org.json.JSONArray;
import org.json.JSONObject;

import java.io.ByteArrayOutputStream;
import java.io.InputStream;

/**
 * 2強レース(個人用)。下の切り替えで「予想」(GitHub Pages のアプリ)と「IPAT」(JRA のネット投票)を行き来する。
 * 予想の「IPATで入力」を押すと、IPAT の画面を順に押して「金額入力」の画面まで進める(途中で画面に触ると止まる)。
 * 押す場所は assets/autopilot.js が画面の文字から決め、このアプリが指で押したのと同じように押す。
 * 金額・セット・投票は自分で入れる。ログイン情報はこのアプリでは持たず、IPAT の画面に入れる。
 */
public class MainActivity extends Activity {
    static final String APP_URL = "https://honda1986.github.io/keeeiba-app/keiba/?app=1";
    static final String IPAT_SP = "https://www.ipat.jra.go.jp/sp/";
    static final String IPAT_PC = "https://www.ipat.jra.go.jp/";
    static final long TICK_MS = 150, LIMIT_MS = 3 * 60 * 1000;   // 0.15秒ごとに画面を見る。3分進まなければ止める

    WebView web, ipat;
    FrameLayout ipatBox;
    TextView betBar;
    Button tabWeb, tabIpat;
    SharedPreferences prefs;
    boolean ipatLoaded = false;

    // 自動入力: task = 1点分の買い目, steps = 押した数, phase = run / final / off
    final Handler handler = new Handler(Looper.getMainLooper());
    JSONObject task;
    String taskText = "";
    int steps = 0;
    String phase = "off";
    long lastAt = 0;
    String lastMsg = "", lastInfo = "", tapInfo = "";   // 止まったときにコピーする手がかり(最後の画面・様子・押した所)
    final java.util.ArrayDeque<String> history = new java.util.ArrayDeque<>();   // 最近の様子(同じものは続けて入れない)
    String autopilot;
    boolean selfTouch = false;                // アプリが押している最中(自分で触ったのと見分ける)

    @Override
    protected void onCreate(Bundle saved) {
        super.onCreate(saved);
        prefs = getSharedPreferences("keiba", MODE_PRIVATE);
        CookieManager.getInstance().setAcceptCookie(true);
        autopilot = readAsset("autopilot.js");

        web = newWebView();
        web.addJavascriptInterface(new Bridge(), "KeibaApp");
        ipat = newWebView();
        CookieManager.getInstance().setAcceptThirdPartyCookies(ipat, true);
        // 自動入力の途中で自分で IPAT の画面に触ったら、自分で続けるとみて自動入力を止める
        ipat.setOnTouchListener((v, e) -> {
            if (!selfTouch && e.getActionMasked() == MotionEvent.ACTION_DOWN && "run".equals(phase)) {
                report("手で止めた所: " + lastMsg);
                stopTask();
                toast("画面に触ったので自動入力を止めました");
            }
            return false;
        });
        // アプリ内の画面(WebView)だと断るサイトがあるので、IPAT にはふつうのスマホのブラウザとして名乗る
        WebSettings is = ipat.getSettings();
        is.setUserAgentString(is.getUserAgentString().replace("; wv", "").replaceAll("Version/[0-9.]+ ", ""));

        // IPAT の上の帯: 自動入力が止まった(うまくいかなかった)ときだけ出す。押すと消える、長押しで買い目をコピー
        betBar = new TextView(this);
        betBar.setTextSize(15);
        betBar.setTypeface(Typeface.DEFAULT_BOLD);
        betBar.setTextColor(Color.WHITE);
        betBar.setBackgroundColor(0xFFC62828);
        betBar.setPadding(dp(14), dp(10), dp(14), dp(10));
        betBar.setVisibility(View.GONE);
        betBar.setOnClickListener(v -> betBar.setVisibility(View.GONE));
        betBar.setOnLongClickListener(v -> {
            copy(taskText);
            return true;
        });

        LinearLayout ipatCol = new LinearLayout(this);
        ipatCol.setOrientation(LinearLayout.VERTICAL);
        ipatCol.addView(betBar, new LinearLayout.LayoutParams(ViewGroup.LayoutParams.MATCH_PARENT, ViewGroup.LayoutParams.WRAP_CONTENT));
        ipatCol.addView(ipat, new LinearLayout.LayoutParams(ViewGroup.LayoutParams.MATCH_PARENT, 0, 1));
        ipatBox = new FrameLayout(this);
        ipatBox.addView(ipatCol);

        FrameLayout pages = new FrameLayout(this);
        pages.addView(web);
        pages.addView(ipatBox);

        LinearLayout bar = new LinearLayout(this);
        bar.setOrientation(LinearLayout.HORIZONTAL);
        tabWeb = tabButton("予想");
        tabIpat = tabButton("IPAT");
        bar.addView(tabWeb, new LinearLayout.LayoutParams(0, dp(52), 1));
        bar.addView(tabIpat, new LinearLayout.LayoutParams(0, dp(52), 1));
        tabWeb.setOnClickListener(v -> {
            if (web.getVisibility() == View.VISIBLE) web.reload();     // 開いているときに押すと読み直す
            show(false);
        });
        tabIpat.setOnClickListener(v -> show(true));
        tabIpat.setOnLongClickListener(v -> {                        // 長押しでスマホ版 / PC版を切り替える
            boolean pc = !prefs.getBoolean("ipat_pc", false);
            prefs.edit().putBoolean("ipat_pc", pc).apply();
            ipat.loadUrl(pc ? IPAT_PC : IPAT_SP);
            ipatLoaded = true;
            show(true);
            toast(pc ? "IPAT を PC版で開きます(自動入力はスマホ版だけ)" : "IPAT をスマホ版で開きます");
            return true;
        });

        LinearLayout root = new LinearLayout(this);
        root.setOrientation(LinearLayout.VERTICAL);
        root.addView(pages, new LinearLayout.LayoutParams(ViewGroup.LayoutParams.MATCH_PARENT, 0, 1));
        root.addView(bar, new LinearLayout.LayoutParams(ViewGroup.LayoutParams.MATCH_PARENT, ViewGroup.LayoutParams.WRAP_CONTENT));
        setContentView(root);

        if (saved != null) {
            web.restoreState(saved);
            Bundle ib = saved.getBundle("ipat");
            if (ib != null && ipat.restoreState(ib) != null) ipatLoaded = true;
        }
        if (web.getUrl() == null) web.loadUrl(APP_URL);
        show(false);
    }

    WebView newWebView() {
        WebView w = new WebView(this);
        WebSettings s = w.getSettings();
        s.setJavaScriptEnabled(true);
        s.setDomStorageEnabled(true);          // 予想アプリの localStorage(投票の記録など)
        s.setDatabaseEnabled(true);
        s.setSupportMultipleWindows(false);
        s.setLoadWithOverviewMode(true);
        s.setUseWideViewPort(true);
        w.setWebChromeClient(new WebChromeClient());   // IPAT の確認ダイアログ(alert / confirm)を出す
        w.setWebViewClient(new WebViewClient() {
            @Override
            public boolean shouldOverrideUrlLoading(WebView view, WebResourceRequest req) {
                Uri u = req.getUrl();
                String h = u.getHost() == null ? "" : u.getHost();
                if (h.endsWith("jra.go.jp") || h.equals("honda1986.github.io")) return false;
                try {
                    startActivity(new Intent(Intent.ACTION_VIEW, u));     // それ以外はブラウザで開く
                } catch (Exception e) {
                    toast("開けません: " + u);
                }
                return true;
            }
        });
        return w;
    }

    Button tabButton(String label) {
        Button b = new Button(this);
        b.setText(label);
        b.setAllCaps(false);
        b.setTextSize(15);
        b.setGravity(Gravity.CENTER);
        return b;
    }

    void show(boolean showIpat) {
        if (showIpat && !ipatLoaded) {
            ipat.loadUrl(prefs.getBoolean("ipat_pc", false) ? IPAT_PC : IPAT_SP);
            ipatLoaded = true;
        }
        web.setVisibility(showIpat ? View.GONE : View.VISIBLE);
        ipatBox.setVisibility(showIpat ? View.VISIBLE : View.GONE);
        tabWeb.setTypeface(showIpat ? Typeface.DEFAULT : Typeface.DEFAULT_BOLD);
        tabIpat.setTypeface(showIpat ? Typeface.DEFAULT_BOLD : Typeface.DEFAULT);
        tabWeb.setAlpha(showIpat ? 0.6f : 1f);
        tabIpat.setAlpha(showIpat ? 1f : 0.6f);
    }

    // ---------------------------------------------------------------- 自動入力
    void startTask(JSONObject t, String text) {
        task = t;
        taskText = text;
        steps = 0;
        lastMsg = "";
        lastInfo = "";
        tapInfo = "";
        history.clear();
        phase = "run";
        lastAt = System.currentTimeMillis();
        betBar.setVisibility(View.GONE);
        copy(text);
        show(true);
        handler.removeCallbacks(tick);
        handler.postDelayed(tick, TICK_MS);
    }

    void stopTask() {
        phase = "off";
        handler.removeCallbacks(tick);
    }

    /** うまくいかなかったときだけ、IPAT の上に赤い帯で知らせる */
    void showError(String msg) {
        betBar.setText("買い目  " + taskText + "\n" + msg + "\n(押すと消える・長押しで買い目をコピー)");
        betBar.setVisibility(View.VISIBLE);
    }

    final Runnable tick = new Runnable() {
        @Override
        public void run() {
            if (!"run".equals(phase) || task == null) return;
            if (System.currentTimeMillis() - lastAt > LIMIT_MS) {
                stopTask();
                report("時間切れ: " + lastMsg);
                showError("時間がかかりすぎたので止めました。手で続けてください\n(止めた所の情報をコピーしました。困ったら貼り付けて送ってください)");
                return;
            }
            if (ipatBox.getVisibility() != View.VISIBLE) {           // 予想を見ているあいだは待つ
                handler.postDelayed(this, TICK_MS);
                return;
            }
            // 中でエラーが出たら、読み込み中(null)と区別できるように error として返す
            String js = "(function(){try{return (" + autopilot + "\n)(" + task.toString() + "," + (steps > 0) + ");}"
                    + "catch(e){return JSON.stringify({st:'error',msg:'自動入力の中でエラーが出ました: '+e.message,"
                    + "info:String(e.stack||'').slice(0,400)});}})()";
            ipat.evaluateJavascript(js, MainActivity.this::onStep);
        }
    };

    void onStep(String raw) {
        if (raw == null || "null".equals(raw)) {                       // 画面の読み込み中
            if ("run".equals(phase)) handler.postDelayed(tick, TICK_MS);
            return;
        }
        try {
            // evaluateJavascript は JSON の文字列を、さらに文字列として返す
            String s = new JSONArray("[" + raw + "]").getString(0);
            JSONObject o = new JSONObject(s);
            String st = o.optString("st"), msg = o.optString("msg"), info = o.optString("info");
            if (!"wait".equals(st) || !msg.isEmpty()) lastMsg = msg;
            if ("tap".equals(st)) tapInfo = info;
            String h = st + " " + msg;
            if (!h.equals(history.peekLast())) {
                history.addLast(h);
                if (history.size() > 10) history.removeFirst();
            }
            lastInfo = "画面 " + o.optString("page") + " | " + (info.isEmpty() ? "最後に押した所 " + tapInfo : info);
            if ("tap".equals(st) || "login".equals(st)) {       // 進んだ(ログイン中も)ら、止めるまでの時間を延ばす
                lastAt = System.currentTimeMillis();
            }
            if ("tap".equals(st)) {
                if (ipatBox.getVisibility() == View.VISIBLE) {     // 予想に切り替えていたら押さない
                    steps++;
                    tapAt((float) o.getDouble("x"), (float) o.getDouble("y"));
                }
            } else if ("final".equals(st)) {          // 金額入力の画面に着いた: ここで止まる(帯は、買い目と違うときだけ)
                handler.removeCallbacks(tick);
                phase = "final";
                if (o.optBoolean("warn")) showError(msg);
                return;
            } else if ("error".equals(st)) {
                stopTask();
                report("止めた所: " + msg);
                showError("止めました: " + msg + "\n(止めた所の情報をコピーしました。困ったら貼り付けて送ってください)");
                return;
            }
        } catch (Exception e) {
            lastMsg = "画面を読めませんでした";      // 読み込み中などで読めないことがある。続けて見る
        }
        if ("run".equals(phase)) handler.postDelayed(tick, TICK_MS);
    }

    /** 止まった所の手がかり(画面の名前・押そうとした所の形。お金や個人の情報は入れない)をコピーする */
    void report(String head) {
        String v = "";
        try {
            v = getPackageManager().getPackageInfo(getPackageName(), 0).versionName;
        } catch (Exception e) {
            // 無視
        }
        copy(head + "\n" + lastInfo + "\n流れ " + String.join(" → ", history)
                + "\n買い目 " + (task == null ? "" : task.toString()) + "\nアプリ " + v);
    }

    /** IPAT の画面を、画面に対する割合 (fx, fy) の所で指で押したのと同じように押す */
    void tapAt(float fx, float fy) {
        float x = fx * ipat.getWidth(), y = fy * ipat.getHeight();
        long t = SystemClock.uptimeMillis();
        MotionEvent down = MotionEvent.obtain(t, t, MotionEvent.ACTION_DOWN, x, y, 0);
        down.setSource(InputDevice.SOURCE_TOUCHSCREEN);
        selfTouch = true;
        ipat.dispatchTouchEvent(down);
        selfTouch = false;
        down.recycle();
        handler.postDelayed(() -> {
            MotionEvent up = MotionEvent.obtain(t, SystemClock.uptimeMillis(), MotionEvent.ACTION_UP, x, y, 0);
            up.setSource(InputDevice.SOURCE_TOUCHSCREEN);
            selfTouch = true;
            ipat.dispatchTouchEvent(up);
            selfTouch = false;
            up.recycle();
        }, 80);
    }

    String readAsset(String name) {
        try (InputStream in = getAssets().open(name)) {
            ByteArrayOutputStream b = new ByteArrayOutputStream();
            byte[] buf = new byte[8192];
            int n;
            while ((n = in.read(buf)) > 0) b.write(buf, 0, n);
            return b.toString("UTF-8");
        } catch (Exception e) {
            return "(function(){return JSON.stringify({st:'error',msg:'自動入力を読めません'})})";
        }
    }

    void copy(String text) {
        ClipboardManager cm = (ClipboardManager) getSystemService(Context.CLIPBOARD_SERVICE);
        cm.setPrimaryClip(ClipData.newPlainText("買い目", text));
    }

    void toast(String s) {
        Toast.makeText(this, s, Toast.LENGTH_SHORT).show();
    }

    int dp(int v) {
        return Math.round(v * getResources().getDisplayMetrics().density);
    }

    /** 予想アプリ(Web)から呼ばれる */
    class Bridge {
        /** json = {text, bet: {venue, dow, rno, type, method, combo}} */
        @JavascriptInterface
        public void buy(String json) {
            runOnUiThread(() -> {
                try {
                    JSONObject o = new JSONObject(json);
                    startTask(o.getJSONObject("bet"), o.optString("text"));
                } catch (Exception e) {
                    toast("買い目を読めません");
                }
            });
        }

        @JavascriptInterface
        public void openIpat() {
            runOnUiThread(() -> show(true));
        }

        @JavascriptInterface
        public String version() {
            try {
                return getPackageManager().getPackageInfo(getPackageName(), 0).versionName;
            } catch (Exception e) {
                return "";
            }
        }
    }

    @Override
    public void onBackPressed() {
        if (ipatBox.getVisibility() == View.VISIBLE) {
            show(false);      // IPAT は「戻る」を使わないように、と案内しているので、IPAT では予想に切り替えるだけ
        } else if (web.canGoBack()) {
            web.goBack();
        } else {
            super.onBackPressed();
        }
    }

    @Override
    protected void onSaveInstanceState(Bundle out) {
        super.onSaveInstanceState(out);
        web.saveState(out);
        Bundle b = new Bundle();
        ipat.saveState(b);
        out.putBundle("ipat", b);
    }

    @Override
    protected void onPause() {
        super.onPause();
        CookieManager.getInstance().flush();
    }
}
