from scq.impressions import ImpressionConfig, simulate_impressions


def test_panel_is_fully_crossed_and_balanced():
    config = ImpressionConfig(n_respondents=20, n_items=30)
    df = simulate_impressions(config)
    assert len(df) == 20 * 30
    assert df.groupby("item_id").size().nunique() == 1
    assert df.groupby("respondent_id").size().nunique() == 1
    counts = df.groupby(["respondent_id", "item_id"]).size()
    assert (counts == 1).all()


def test_item_effect_is_shared_between_usefulness_and_engagement():
    """The correlation between usefulness and engagement must come from
    the SAME per-item true_item_effect column, not from two independently
    drawn effects: regressing engagement on true_item_effect alone (across
    items) must show a nonzero, positive relationship.
    """
    df = simulate_impressions(ImpressionConfig(n_respondents=50, n_items=60))
    item_level = df.groupby("item_id").agg(
        true_item_effect=("true_item_effect", "first"),
        engagement_score=("engagement_score", "mean"),
    )
    corr = item_level["true_item_effect"].corr(item_level["engagement_score"])
    assert corr > 0.3


def test_respondent_effect_does_not_leak_into_engagement():
    """Engagement is constructed without any respondent_effect term; a
    respondent's AVERAGE engagement across all items they rated must not
    depend on that respondent's true_respondent_effect (usefulness bias),
    beyond sampling noise.
    """
    df = simulate_impressions(ImpressionConfig(n_respondents=50, n_items=60))
    resp_level = df.groupby("respondent_id").agg(
        true_respondent_effect=("true_respondent_effect", "first"),
        engagement_score=("engagement_score", "mean"),
    )
    corr = resp_level["true_respondent_effect"].corr(resp_level["engagement_score"])
    assert abs(corr) < 0.3
