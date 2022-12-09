def getfields(windows,rolling_windows):
    fields = []
    names = []
    hf = False
    if hf:
        template_if = "If(IsNull({1}), {0}, {1})"
        template_paused = "{0}"
        template_fillnan = "BFillNan(FFillNan({0}))"
        # Because there is no vwap field in the yahoo data, a method similar to Simpson integration is used to approximate vwap
        simpson_vwap = "($open + 2*$high + 2*$low + $close)/6"

        def get_normalized_price_feature(price_field, shift=0):
            """Get normalized price feature ops"""
            if shift == 0:
                template_norm = "Cut({0}/Ref(DayLast({1}), 240), 240, None)"
            else:
                template_norm = "Cut(Ref({0}, " + str(shift) + ")/Ref(DayLast({1}), 240), 240, None)"

            feature_ops = template_norm.format(
                template_if.format(
                    template_fillnan.format(template_paused.format("$close")),
                    template_paused.format(price_field),
                ),
                template_fillnan.format(template_paused.format("$close")),
            )
            return feature_ops

        fields += [get_normalized_price_feature("$open", 0)]
        fields += [get_normalized_price_feature("$high", 0)]
        fields += [get_normalized_price_feature("$low", 0)]
        fields += [get_normalized_price_feature("$close", 0)]
        fields += [get_normalized_price_feature(simpson_vwap, 0)]
        names += ["$open", "$high", "$low", "$close", "$vwap"]

        fields += [get_normalized_price_feature("$open", 240)]
        fields += [get_normalized_price_feature("$high", 240)]
        fields += [get_normalized_price_feature("$low", 240)]
        fields += [get_normalized_price_feature("$close", 240)]
        fields += [get_normalized_price_feature(simpson_vwap, 240)]
        names += ["$open_1", "$high_1", "$low_1", "$close_1", "$vwap_1"]

        fields += [
            "Cut({0}/Ref(DayLast(Mean({0}, 7200)), 240), 240, None)".format(
                "If(IsNull({0}), 0, If(Or(Gt({1}, Mul(1.001, {3})), Lt({1}, Mul(0.999, {2}))), 0, {0}))".format(
                    template_paused.format("$volume"),
                    template_paused.format(simpson_vwap),
                    template_paused.format("$low"),
                    template_paused.format("$high"),
                )
            )
        ]
        names += ["$volume"]
        fields += [
            "Cut(Ref({0}, 240)/Ref(DayLast(Mean({0}, 7200)), 240), 240, None)".format(
                "If(IsNull({0}), 0, If(Or(Gt({1}, Mul(1.001, {3})), Lt({1}, Mul(0.999, {2}))), 0, {0}))".format(
                    template_paused.format("$volume"),
                    template_paused.format(simpson_vwap),
                    template_paused.format("$low"),
                    template_paused.format("$high"),
                )
            )
        ]
        names += ["$volume_1"]

        



    fields += [
        "($close-$open)/$open",
        "($high-$low)/$open",
        "($close-$open)/($high-$low+1e-12)",
        "($high-Greater($open, $close))/$open",
        "($high-Greater($open, $close))/($high-$low+1e-12)",
        "(Less($open, $close)-$low)/$open",
        "(Less($open, $close)-$low)/($high-$low+1e-12)",
        "(2*$close-$high-$low)/$open",
        "(2*$close-$high-$low)/($high-$low+1e-12)",
    ]
    names += [
        "KMID",
        "KLEN",
        "KMID2",
        "KUP",
        "KUP2",
        "KLOW",
        "KLOW2",
        "KSFT",
        "KSFT2",
    ]
    feature = ["OPEN", "HIGH", "LOW", "CLOSE", "VWAP"]
    for field in feature:
        field = field.lower()
        fields += [
            "Ref($%s, %d)/$close" % (field, d) if d != 0 else "$%s/$close" % field
            for d in windows
        ]
        names += [field.upper() + str(d) for d in windows]
    fields += [
        "Ref($volume, %d)/($volume+1e-12)" % d if d != 0 else "$volume/($volume+1e-12)"
        for d in windows
    ]
    names += ["VOLUME" + str(d) for d in windows]
    fields += [
        "((Ref($open,%d) + Ref($close,%d))/2)/Ref($close, %d)" % (d + 1, d + 1, d)
        if d != 0
        else "((Ref($open,%d) + Ref($close,%d))/2)/$close" % (d + 1, d + 1)
        for d in windows
    ]
    names += ["HEIASHO" + str(d) for d in windows]
    fields += [
        "((Ref($open,%d) + Ref($close,%d))/2)/$close" % (d + 1, d + 1) for d in windows
    ]
    names += ["HEIASHON" + str(d) for d in windows]
    fields += [
        "((Ref($adj, %d)*3/4) + (Ref($close, %d)*1/4))/Ref($close, %d)"
        % (d, d, d)
        if d != 0
        else "(($adj*3/4) + ($close*1/4))/$close"
        for d in windows
    ]
    names += ["HEIASHC" + str(d) for d in windows]
    fields += [
        "((Ref($adj, %d)*3/4) + (Ref($close, %d)*1/4))/$close"
        % (d, d)
        if d != 0
        else "(($adj*3/4) + ($close*1/4))/$close"
        for d in windows
    ]
    names += ["HEIASHCN" + str(d) for d in windows]
    windows = rolling_windows
    fields += ["SlopeBTC($close, %d)/RefBTC($close,0)" % d for d in windows]
    names += ["SLPBTC%d" % d for d in windows]
    fields += ["SlopeETH($close, %d)/RefETH($close,0)" % d for d in windows]
    names += ["SLPETH%d" % d for d in windows]
    fields += ["DeltaBTC($close, %d)/RefBTC($close,0)" % d for d in windows]
    names += ["DELBTC%d" % d for d in windows]
    fields += ["DeltaETH($close, %d)/RefETH($close,0)" % d for d in windows]
    names += ["DELETH%d" % d for d in windows]
    fields += ["CorrBTC($close, $volume, %d)" % d for d in windows]
    names += ["CORRBTCV%d" % d for d in windows]
    fields += ["CorrETH($close, $volume, %d)" % d for d in windows]
    names += ["CORRETHV%d" % d for d in windows]
    fields += ["CorrBTC($close, $close, %d)" % d for d in windows]
    names += ["CORRBTC%d" % d for d in windows]
    fields += ["CorrETH($close, $close, %d)" % d for d in windows]
    names += ["CORRETH%d" % d for d in windows]
    fields += ["CorrBTC($volume, $volume, %d)" % d for d in windows]
    names += ["CORRBTCVV%d" % d for d in windows]
    fields += ["CorrETH($volume, $volume, %d)" % d for d in windows]
    names += ["CORRETHVV%d" % d for d in windows]
    fields += ["((Max($high, %d)- $high)/$high)*100" % d for d in windows]
    names += ["MAXGE%d" % d for d in windows]
    fields += ["((Min($low, %d) - $low)/$low)*100" % d for d in windows]
    names += ["MINGE%d" % d for d in windows]
    fields += ["((Max($high, %d) - $close)/$close)*100" % d for d in windows]
    names += ["MAXCGE%d" % d for d in windows]
    fields += ["((Min($low, %d) - $close)/$close)*100" % d for d in windows]
    names += ["MINCGE%d" % d for d in windows]
    # STOCHASTIC STUFF
    fields += [
        "100*(($close - Min($low,%d))/(Max($high,%d) - Min($low,%d)))" % (d, d, d)
        for d in windows
    ]
    names += ["STOCH%d" % d for d in windows]
    fields += [
        "Mean(100*(($close - Min($low,%d))/(Max($high,%d) - Min($low,%d))),%d)"
        % (d, d, d, d)
        for d in windows
    ]
    names += ["STOCHEMA%d" % d for d in windows]
    fields += [
        "Mean(100*(($close - Min($low,%d))/(Max($high,%d) - Min($low,%d))),%d) - 100*(($close - Min($low,%d))/(Max($high,%d) - Min($low,%d)))"
        % (d, d, d, d, d, d, d)
        for d in windows
    ]
    names += ["STOCHEMAD%d" % d for d in windows]
    # #EMA/MACD
    fields += ["(EMA($close,12) - EMA($close,26))/$close"]
    names += ["MACDs"]
    fields += ["EMA((EMA($close,12) - EMA($close,26)),9)/$close"]
    names += ["MACDF"]
    fields += [
        "(EMA((EMA($close,12) - EMA($close,26)),9) - EMA($close,12) - EMA($close,26))/$close"
    ]
    names += ["MACDI"]
    # #BOLLINGER
    fields += [
        "((Mean($adj,%d) + 2*Std($adj,%d)) - $close)/$close"
        % (d, d)
        for d in windows
    ]
    names += ["BOLU%d" % d for d in windows]
    fields += [
        "((Mean($adj,%d) - 2*Std($adj,%d)) - $close)/$close"
        % (d, d)
        for d in windows
    ]
    names += ["BOLD%d" % d for d in windows]
    # RSI
    fields += [
        "100 - (100/(1 + Mean(If($close - Ref($close,1)>0,$close,np.nan),%d)/Mean(If($close - Ref($close,1)<0,$close,np.nan),%d)))"
        % (d, d)
        for d in windows
    ]
    names += ["RSI%d" % d for d in windows]
    # MINMAX
    # fields += ["((Max(RefBTC($high,1), %d)- RefBTC($high,0))/RefBTC($high,0))*100" % d for d in windows]
    # names += ["MAXGEBTC%d" % d for d in windows]
    # fields += ["((Min(RefBTC($low,1), %d) -RefBTC($low,0))/RefBTC($low,0))*100" % d for d in windows]
    # names += ["MINGEBTC%d" % d for d in windows]
    # fields += ["((Max(RefBTC($high,1), %d) - RefBTC($close,0))/RefBTC($close,0))*100" % d for d in windows]
    # names += ["MAXCGEBTC%d" % d for d in windows]
    # fields += ["((Min(RefBTC($low,1), %d) - RefBTC($close,0))/RefBTC($close,0))*100" % d for d in windows]
    # names += ["MINCGEBTC%d" % d for d in windows]
    # fields += ["((Max(RefETH($high,1), %d)- RefETH($high,0))/RefETH($high,0))*100" % d for d in windows]
    # names += ["MAXGEETH%d" % d for d in windows]
    # fields += ["((Min(RefETH($low,1), %d) -RefETH($low,0))/RefETH($low,0))*100" % d for d in windows]
    # names += ["MINGEETH%d" % d for d in windows]
    # fields += ["((Max(RefETH($high,1), %d) - RefETH($close,0))/RefETH($close,0))*100" % d for d in windows]
    # names += ["MAXCGEETH%d" % d for d in windows]
    # fields += ["((Min(RefETH($low,1), %d) - RefETH($close,0))/RefETH($close,0))*100" % d for d in windows]
    # names += ["MINCGEETH%d" % d for d in windows]
    fields += ["Ref($close, %d)/$close" % d for d in windows]
    names += ["ROC%d" % d for d in windows]
    fields += ["Mean($close, %d)/$close" % d for d in windows]
    names += ["MA%d" % d for d in windows]
    fields += ["Std($close, %d)/$close" % d for d in windows]
    names += ["STD%d" % d for d in windows]
    fields += ["Slope($close, %d)/$close" % d for d in windows]
    names += ["BETA%d" % d for d in windows]
    fields += ["Rsquare($close, %d)" % d for d in windows]
    names += ["RSQR%d" % d for d in windows]
    fields += ["Resi($close, %d)/$close" % d for d in windows]
    names += ["RESI%d" % d for d in windows]
    fields += ["Max($high, %d)/$close" % d for d in windows]
    names += ["MAX%d" % d for d in windows]
    fields += ["Min($low, %d)/$close" % d for d in windows]
    names += ["MIN%d" % d for d in windows]
    fields += ["Quantile($close, %d, 0.8)/$close" % d for d in windows]
    names += ["QTLU%d" % d for d in windows]
    fields += ["Quantile($close, %d, 0.2)/$close" % d for d in windows]
    names += ["QTLD%d" % d for d in windows]
    # fields += ["Rank($close, %d)" % d for d in windows]
    # names += ["RANK%d" % d for d in windows]
    fields += [
        "($close-Min($low, %d))/(Max($high, %d)-Min($low, %d)+1e-12)" % (d, d, d)
        for d in windows
    ]
    names += ["RSV%d" % d for d in windows]
    fields += ["IdxMax($high, %d)/%d" % (d, d) for d in windows]
    names += ["IMAX%d" % d for d in windows]
    fields += ["IdxMin($low, %d)/%d" % (d, d) for d in windows]
    names += ["IMIN%d" % d for d in windows]
    fields += ["(IdxMax($high, %d)-IdxMin($low, %d))/%d" % (d, d, d) for d in windows]
    names += ["IMXD%d" % d for d in windows]
    fields += ["Corr($close, Log($volume+1), %d)" % d for d in windows]
    names += ["CORR%d" % d for d in windows]
    fields += [
        "Corr($close/Ref($close,1), Log($volume/Ref($volume, 1)+1), %d)" % d
        for d in windows
    ]
    names += ["CORD%d" % d for d in windows]
    fields += ["Mean($close>Ref($close, 1), %d)" % d for d in windows]
    names += ["CNTP%d" % d for d in windows]
    fields += ["Mean($close<Ref($close, 1), %d)" % d for d in windows]
    names += ["CNTN%d" % d for d in windows]
    fields += [
        "Mean($close>Ref($close, 1), %d)-Mean($close<Ref($close, 1), %d)" % (d, d)
        for d in windows
    ]
    names += ["CNTD%d" % d for d in windows]
    fields += [
        "Sum(Greater($close-Ref($close, 1), 0), %d)/(Sum(Abs($close-Ref($close, 1)), %d)+1e-12)"
        % (d, d)
        for d in windows
    ]
    names += ["SUMP%d" % d for d in windows]
    fields += [
        "Sum(Greater(Ref($close, 1)-$close, 0), %d)/(Sum(Abs($close-Ref($close, 1)), %d)+1e-12)"
        % (d, d)
        for d in windows
    ]
    names += ["SUMN%d" % d for d in windows]
    fields += [
        "(Sum(Greater($close-Ref($close, 1), 0), %d)-Sum(Greater(Ref($close, 1)-$close, 0), %d))"
        "/(Sum(Abs($close-Ref($close, 1)), %d)+1e-12)" % (d, d, d)
        for d in windows
    ]
    names += ["SUMD%d" % d for d in windows]
    fields += ["Mean($volume, %d)/($volume+1e-12)" % d for d in windows]
    names += ["VMA%d" % d for d in windows]
    fields += ["Std($volume, %d)/($volume+1e-12)" % d for d in windows]
    names += ["VSTD%d" % d for d in windows]
    fields += [
        "Std(Abs($close/Ref($close, 1)-1)*$volume, %d)/(Mean(Abs($close/Ref($close, 1)-1)*$volume, %d)+1e-12)"
        % (d, d)
        for d in windows
    ]
    names += ["WVMA%d" % d for d in windows]
    fields += [
        "Sum(Greater($volume-Ref($volume, 1), 0), %d)/(Sum(Abs($volume-Ref($volume, 1)), %d)+1e-12)"
        % (d, d)
        for d in windows
    ]
    names += ["VSUMP%d" % d for d in windows]
    fields += [
        "Sum(Greater(Ref($volume, 1)-$volume, 0), %d)/(Sum(Abs($volume-Ref($volume, 1)), %d)+1e-12)"
        % (d, d)
        for d in windows
    ]
    names += ["VSUMN%d" % d for d in windows]
    fields += [
        "(Sum(Greater($volume-Ref($volume, 1), 0), %d)-Sum(Greater(Ref($volume, 1)-$volume, 0), %d))"
        "/(Sum(Abs($volume-Ref($volume, 1)), %d)+1e-12)" % (d, d, d)
        for d in windows
    ]
    names += ["VSUMD%d" % d for d in windows]
    return fields, names
