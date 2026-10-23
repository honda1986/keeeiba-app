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
import android.view.Gravity;
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

import org.json.JSONObject;

/**
 * 2強レース(個人用)。下の切り替えで「予想」(GitHub Pages のアプリ)と「IPAT」(JRA のネット投票)を行き来する。
 * 馬券は IPAT の画面で自分で買う(自動では買わない)。ログイン情報はこのアプリでは持たず、IPAT の画面に入れる。
 * 予想の投票タブで「IPATで買う」を押すと、買い目をコピーして IPAT に切り替え、上に買い目を出しておく。
 */
public class MainActivity extends Activity {
    static final String APP_URL = "https://honda1986.github.io/keeeiba-app/keiba/?app=1";
    static final String IPAT_SP = "https://www.ipat.jra.go.jp/sp/";
    static final String IPAT_PC = "https://www.ipat.jra.go.jp/";

    WebView web, ipat;
    FrameLayout ipatBox;
    TextView betBar;
    Button tabWeb, tabIpat;
    SharedPreferences prefs;
    boolean ipatLoaded = false;

    @Override
    protected void onCreate(Bundle saved) {
        super.onCreate(saved);
        prefs = getSharedPreferences("keiba", MODE_PRIVATE);
        CookieManager.getInstance().setAcceptCookie(true);

        web = newWebView();
        web.addJavascriptInterface(new Bridge(), "KeibaApp");
        ipat = newWebView();
        CookieManager.getInstance().setAcceptThirdPartyCookies(ipat, true);

        // IPAT の上に出す買い目(押すと消える、長押しでもう一度コピー)
        betBar = new TextView(this);
        betBar.setTextSize(15);
        betBar.setTypeface(Typeface.DEFAULT_BOLD);
        betBar.setTextColor(Color.WHITE);
        betBar.setBackgroundColor(0xFF1F6FEB);
        betBar.setPadding(dp(14), dp(10), dp(14), dp(10));
        betBar.setVisibility(View.GONE);
        betBar.setOnClickListener(v -> betBar.setVisibility(View.GONE));
        betBar.setOnLongClickListener(v -> {
            copy(String.valueOf(betBar.getTag()));
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
            toast(pc ? "IPAT を PC版で開きます" : "IPAT をスマホ版で開きます");
            return true;
        });

        LinearLayout root = new LinearLayout(this);
        root.setOrientation(LinearLayout.VERTICAL);
        root.addView(pages, new LinearLayout.LayoutParams(ViewGroup.LayoutParams.MATCH_PARENT, 0, 1));
        root.addView(bar, new LinearLayout.LayoutParams(ViewGroup.LayoutParams.MATCH_PARENT, ViewGroup.LayoutParams.WRAP_CONTENT));
        setContentView(root);

        if (saved != null) {
            web.restoreState(saved);
            ipat.restoreState(saved.getBundle("ipat") != null ? saved.getBundle("ipat") : new Bundle());
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

    void copy(String text) {
        ClipboardManager cm = (ClipboardManager) getSystemService(Context.CLIPBOARD_SERVICE);
        cm.setPrimaryClip(ClipData.newPlainText("買い目", text));
        toast("コピーしました: " + text);
    }

    void toast(String s) {
        Toast.makeText(this, s, Toast.LENGTH_SHORT).show();
    }

    int dp(int v) {
        return Math.round(v * getResources().getDisplayMetrics().density);
    }

    /** 予想アプリ(Web)から呼ばれる */
    class Bridge {
        @JavascriptInterface
        public void buy(String json) {
            runOnUiThread(() -> {
                try {
                    JSONObject o = new JSONObject(json);
                    String text = o.optString("text");
                    betBar.setTag(text);
                    betBar.setText("買い目  " + text + "\n(押すと消える・長押しでもう一度コピー)");
                    betBar.setVisibility(View.VISIBLE);
                    copy(text);
                    show(true);
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
        WebView w = ipatBox.getVisibility() == View.VISIBLE ? ipat : web;
        if (w.canGoBack()) {
            w.goBack();
        } else if (w == ipat) {
            show(false);
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
