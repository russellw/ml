bool match(term a, term b, vec<pair<term, term>> &r);
bool unify(term a, term b, vec<pair<term, term>> &r);
term replace(term a, const vec<pair<term, term>> &unified);
