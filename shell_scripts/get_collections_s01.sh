curl -s http://200.144.192.67:3000/api/v1/knowledge/ \
  -H "Authorization: Bearer sk-cbca990e6da14666ab7f2291d2fd1669" \
  | jq -r '.[] '
